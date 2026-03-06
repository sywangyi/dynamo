# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoTokenizer

import dynamo.nixl_connect as connect
from dynamo._core import Client, Context
from dynamo.sglang.args import Config
from dynamo.sglang.multimodal_utils import (
    multimodal_request_to_sglang,
    process_sglang_stream_response,
)
from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    MultiModalRequest,
    SglangMultimodalRequest,
)
from dynamo.sglang.request_handlers.handler_base import BaseGenerativeHandler

logger = logging.getLogger(__name__)


class MultimodalProcessorHandler(BaseGenerativeHandler):
    """
    Handler for multimodal processor component that processes multimodal requests
    and forwards them to the encode worker.
    """

    def __init__(
        self,
        config: Config,
        encode_worker_client: Client,
        pd_worker_client: Client,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(config)
        self.encode_worker_client = encode_worker_client
        self.pd_worker_client = pd_worker_client
        self.chat_template = getattr(config.server_args, "chat_template", "qwen2-vl")
        self.model = config.server_args.model_path
        self.shutdown_event = shutdown_event
        self.split_encode = os.getenv("DYN_SPLIT_ENCODE", "0") == "1"
        self.split_encode_cpu_ratio = self._parse_cpu_ratio(
            os.getenv("DYN_SPLIT_ENCODE_CPU_RATIO", "0")
        )
        self._encoder_inflight: dict[int, int] = defaultdict(int)
        self._encoder_device: dict[int, str] = {}
        self._encoder_route_lock = asyncio.Lock()
        self._encoder_probe_lock = asyncio.Lock()
        self._split_metrics_lock = asyncio.Lock()
        self._encoder_rr_index = 0
        self._total_requests = 0
        self._split_attempt_requests = 0
        self._single_path_requests = 0
        self._connector = connect.Connector()

        # Initialize tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
            truncation_side="left",
        )

        image_token_str = chat_templates[self.chat_template].copy().image_token
        if image_token_str == "<|vision_start|><|image_pad|><|vision_end|>":
            self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        else:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token_str)

    @staticmethod
    def _parse_cpu_ratio(raw: str) -> float:
        """Parse CPU split ratio from env.

        Accepts either fraction form (0.25) or percent form (25).
        Values are clamped to [0.0, 1.0].
        """
        try:
            ratio = float(raw)
        except (TypeError, ValueError):
            return 0.0

        if ratio > 1.0:
            ratio = ratio / 100.0
        return max(0.0, min(1.0, ratio))

    def cleanup(self):
        pass

    @staticmethod
    def _parse_response_payload(resp):
        if hasattr(resp, "data"):
            raw_data = resp.data
            if callable(raw_data):
                raw_data = raw_data()
            return raw_data
        return resp

    def _expand_image_placeholders(
        self, token_ids: list[int], token_counts: list[int]
    ) -> list[int]:
        expanded = token_ids[:]
        search_start = 0
        for num_image_tokens in token_counts:
            try:
                image_token_id_index = expanded.index(self.image_token_id, search_start)
            except ValueError as e:
                raise ValueError(
                    "Not enough image tokens found for provided images"
                ) from e

            expanded = (
                expanded[:image_token_id_index]
                + [self.image_token_id] * int(num_image_tokens)
                + expanded[image_token_id_index + 1 :]
            )
            search_start = image_token_id_index + int(num_image_tokens)

        return expanded

    @staticmethod
    def _infer_token_counts_from_grids(
        image_grid_thw_list: list[list[int]], total_tokens: int
    ) -> list[int]:
        if total_tokens <= 0:
            raise ValueError("Invalid token statistics for embeddings")

        grid_sizes = []
        for image_grid_thw in image_grid_thw_list:
            if not isinstance(image_grid_thw, list) or len(image_grid_thw) != 3:
                raise ValueError("Cannot split embeddings: invalid image_grid_thw")
            grid_sizes.append(int(image_grid_thw[1] * image_grid_thw[2]))

        total_grid_tokens = sum(grid_sizes)
        if total_grid_tokens <= 0:
            raise ValueError("Invalid grid statistics for embeddings")

        if total_grid_tokens % total_tokens != 0:
            raise ValueError(
                "Cannot infer merge factor: grid token total is not divisible by embedding token total"
            )

        merge_factor = total_grid_tokens // total_tokens
        token_counts = []
        for grid_count in grid_sizes:
            if grid_count % merge_factor != 0:
                raise ValueError(
                    "Cannot split embeddings: per-image grid token count not divisible by inferred merge factor"
                )
            token_counts.append(grid_count // merge_factor)

        if sum(token_counts) != total_tokens:
            raise ValueError(
                "Cannot split embeddings: per-image token counts do not match embedding token total"
            )

        return token_counts

    def _can_split_encode(
        self,
        multimodal_groups: list[MultiModalGroup],
        valid_encode_instances: int,
    ) -> bool:
        if (
            not self.split_encode
            or len(multimodal_groups) <= 1
            or valid_encode_instances <= 1
        ):
            return False
        return all(
            group.multimodal_input is not None
            and group.multimodal_input.image_url is not None
            and group.multimodal_input.video_url is None
            for group in multimodal_groups
        )

    async def _count_valid_encode_instances(self) -> int:
        instances = self.encode_worker_client.instance_ids()
        if not instances:
            return 0

        await self._probe_encoder_devices(instances)

        async with self._encoder_route_lock:
            inflight_snapshot = dict(self._encoder_inflight)

        return sum(
            1
            for instance in instances
            if (
                (self._encoder_device.get(instance) == "none-cpu")
                or (
                    self._encoder_device.get(instance) == "cpu"
                    and inflight_snapshot.get(instance, 0) == 0
                )
            )
        )

    async def _has_idle_cpu_encode_instance(self) -> bool:
        instances = self.encode_worker_client.instance_ids()
        if not instances:
            return False

        await self._probe_encoder_devices(instances)

        async with self._encoder_route_lock:
            inflight_snapshot = dict(self._encoder_inflight)

        return any(
            self._encoder_device.get(instance) == "cpu"
            and inflight_snapshot.get(instance, 0) == 0
            for instance in instances
        )

    async def _record_request_path(self, request_id: str, path: str) -> None:
        async with self._split_metrics_lock:
            self._total_requests += 1
            if path == "split":
                self._split_attempt_requests += 1
            else:
                self._single_path_requests += 1

            logger.info(
                "encode path metrics: request=%s path=%s total=%d split_attempt=%d single=%d",
                request_id,
                path,
                self._total_requests,
                self._split_attempt_requests,
                self._single_path_requests,
            )

    async def _generate_split(
        self,
        request_id: str,
        raw_request: MultiModalRequest,
        sglang_request,
        multimodal_groups: list[MultiModalGroup],
    ):
        encoded_groups: list[MultiModalGroup] = []
        encoded_tensors: list[torch.Tensor] = []
        image_token_counts: list[int] = []

        instances = self.encode_worker_client.instance_ids()
        if not instances:
            instances = await self.encode_worker_client.wait_for_instances()
        await self._probe_encoder_devices(instances)

        async with self._encoder_route_lock:
            inflight_snapshot = dict(self._encoder_inflight)

        cpu_instances = [
            instance
            for instance in instances
            if self._encoder_device.get(instance) == "cpu"
            and inflight_snapshot.get(instance, 0) == 0
        ]
        non_cpu_instances = [
            instance
            for instance in instances
            if self._encoder_device.get(instance) == "none-cpu"
        ]

        total_images = len(multimodal_groups)
        cpu_target = math.floor(total_images * self.split_encode_cpu_ratio)
        if not cpu_instances:
            cpu_target = 0
        if not non_cpu_instances:
            cpu_target = total_images

        logger.debug(
            "split encode plan: request=%s total_images=%d cpu_ratio=%.3f cpu_target=%d non_cpu_target=%d",
            request_id,
            total_images,
            self.split_encode_cpu_ratio,
            cpu_target,
            total_images - cpu_target,
        )
        split_start_time = time.perf_counter()

        async def _encode_batch(
            src_groups: list[MultiModalGroup],
            *,
            prefer_device: str,
            avoid_device: str | None = None,
            expected_count: int,
            batch_name: str,
        ) -> tuple[list[MultiModalGroup], torch.Tensor, list[int], float]:
            batch_groups = [
                MultiModalGroup(
                    multimodal_input=MultiModalInput(
                        image_url=group.multimodal_input.image_url
                    )
                )
                for group in src_groups
            ]
            request_copy = deepcopy(sglang_request)
            encode_request = SglangMultimodalRequest(
                request=request_copy,
                multimodal_inputs=batch_groups,
                encode_only=True,
            )
            encode_request.request.token_ids = [self.image_token_id] * len(batch_groups)

            response_generator, selected_instance = await self._dispatch_to_encoder(
                encode_request.model_dump_json(),
                prefer_device=prefer_device,
                avoid_device=avoid_device,
            )
            batch_start_time = time.perf_counter()
            try:
                first = await anext(response_generator)
                payload = self._parse_response_payload(first)
                if isinstance(payload, str):
                    encoded = SglangMultimodalRequest.model_validate_json(payload)
                else:
                    encoded = SglangMultimodalRequest.model_validate(payload)
            finally:
                if selected_instance is not None:
                    await self._on_encoder_request_done(selected_instance)

            if encoded.serialized_request is None or encoded.embeddings_shape is None:
                raise RuntimeError("encode worker did not return embeddings metadata")
            if not encoded.multimodal_inputs:
                raise RuntimeError("encode worker did not return multimodal groups")
            if len(encoded.multimodal_inputs) != expected_count:
                raise RuntimeError(
                    f"encode worker returned unexpected multimodal group count for {batch_name} batch"
                )

            embedding = torch.empty(
                encoded.embeddings_shape,
                dtype=torch.float16,
                device="cpu",
            )
            descriptor = connect.Descriptor(embedding)
            read_op = await self._connector.begin_read(
                encoded.serialized_request, descriptor
            )
            await read_op.wait_for_completion()

            grids = [group.image_grid_thw for group in encoded.multimodal_inputs]
            if any(grid is None for grid in grids):
                raise RuntimeError("encode worker did not return image_grid_thw")
            token_counts = self._infer_token_counts_from_grids(
                grids,
                embedding.shape[0],
            )

            batch_latency_ms = (time.perf_counter() - batch_start_time) * 1000.0
            logger.info(
                "split encode batch done: request=%s batch=%s images=%d prefer_device=%s selected_instance=%s latency_ms=%.2f",
                request_id,
                batch_name,
                len(batch_groups),
                prefer_device,
                selected_instance,
                batch_latency_ms,
            )

            return encoded.multimodal_inputs, embedding, token_counts, batch_latency_ms

        cpu_groups = multimodal_groups[:cpu_target]
        non_cpu_groups = multimodal_groups[cpu_target:]

        def _split_into_subbatches(
            groups: list[MultiModalGroup], max_parallelism: int
        ) -> list[list[MultiModalGroup]]:
            if not groups:
                return []

            parallelism = max(1, min(len(groups), max_parallelism))
            base_size, remainder = divmod(len(groups), parallelism)

            subbatches: list[list[MultiModalGroup]] = []
            cursor = 0
            for i in range(parallelism):
                size = base_size + (1 if i < remainder else 0)
                next_cursor = cursor + size
                subbatches.append(groups[cursor:next_cursor])
                cursor = next_cursor

            return subbatches

        cpu_parallelism = len(cpu_instances) if cpu_instances else 1
        non_cpu_parallelism = len(non_cpu_instances) if non_cpu_instances else 1

        cpu_subbatches = _split_into_subbatches(cpu_groups, cpu_parallelism)
        non_cpu_subbatches = _split_into_subbatches(non_cpu_groups, non_cpu_parallelism)

        batch_tasks = []

        for i, subbatch in enumerate(cpu_subbatches):
            batch_tasks.append(
                asyncio.create_task(
                    _encode_batch(
                        subbatch,
                        prefer_device="cpu",
                        avoid_device="none-cpu",
                        expected_count=len(subbatch),
                        batch_name=f"cpu-{i}",
                    )
                )
            )

        for i, subbatch in enumerate(non_cpu_subbatches):
            batch_tasks.append(
                asyncio.create_task(
                    _encode_batch(
                        subbatch,
                        prefer_device="none-cpu",
                        avoid_device="cpu",
                        expected_count=len(subbatch),
                        batch_name=f"non-cpu-{i}",
                    )
                )
            )

        batch_results = await asyncio.gather(*batch_tasks)
        for groups_result, embedding_result, token_counts_result, _ in batch_results:
            encoded_groups.extend(groups_result)
            encoded_tensors.append(embedding_result)
            image_token_counts.extend(token_counts_result)

        split_encode_latency_ms = (time.perf_counter() - split_start_time) * 1000.0
        logger.info(
            "split encode merge-ready: request=%s total_images=%d cpu_images=%d non_cpu_images=%d cpu_batches=%d non_cpu_batches=%d total_latency_ms=%.2f",
            request_id,
            total_images,
            len(cpu_groups),
            len(non_cpu_groups),
            len(cpu_subbatches),
            len(non_cpu_subbatches),
            split_encode_latency_ms,
        )

        merged_multimodal_groups: list[MultiModalGroup] = []
        if len(multimodal_groups) != len(encoded_groups):
            raise RuntimeError(
                "split encode merge mismatch: encoded group count does not match source group count"
            )
        for encoded_group in encoded_groups:
            merged_group = MultiModalGroup(
                multimodal_input=MultiModalInput(image_url=None, video_url=None),
                image_grid_thw=encoded_group.image_grid_thw,
            )
            merged_multimodal_groups.append(merged_group)

        merged_embeddings = (
            torch.cat(encoded_tensors, dim=0)
            if len(encoded_tensors) > 1
            else encoded_tensors[0]
        )

        merged_request = SglangMultimodalRequest(
            request=sglang_request,
            multimodal_inputs=merged_multimodal_groups,
        )
        merged_request.request.token_ids = self._expand_image_placeholders(
            merged_request.request.token_ids,
            image_token_counts,
        )
        merged_request.embeddings_shape = tuple(merged_embeddings.shape)
        merged_request.serialized_request = None

        descriptor = connect.Descriptor(merged_embeddings)
        with await self._connector.create_readable(descriptor) as readable:
            merged_request.serialized_request = readable.metadata()
            response_generator = await self.pd_worker_client.round_robin(
                merged_request.model_dump_json()
            )

            finished_sent = False
            accumulated_text = ""
            first_upstream_chunk_time: float | None = None
            first_output_chunk_time: float | None = None
            upstream_chunk_count = 0
            output_chunk_count = 0
            async for resp in response_generator:
                upstream_chunk_count += 1
                if first_upstream_chunk_time is None:
                    first_upstream_chunk_time = time.perf_counter()
                try:
                    raw_data = self._parse_response_payload(resp)
                    if isinstance(raw_data, str):
                        try:
                            response_data = json.loads(raw_data)
                        except json.JSONDecodeError:
                            response_data = {"text": raw_data, "finished": False}
                    else:
                        response_data = raw_data

                    (
                        text_content,
                        accumulated_text,
                        is_finished,
                    ) = process_sglang_stream_response(
                        response_data, self.tokenizer, accumulated_text
                    )

                    if text_content or is_finished:
                        choice: Dict[str, Any] = {
                            "index": 0,
                            "delta": {},
                            "finish_reason": None,
                        }
                        delta: Dict[str, str] = choice["delta"]

                        if text_content and not finished_sent:
                            delta["role"] = "assistant"

                        if text_content:
                            delta["content"] = text_content

                        if is_finished:
                            choice["finish_reason"] = response_data.get(
                                "finish_reason", "stop"
                            )
                            if not finished_sent and not text_content:
                                delta["role"] = "assistant"

                        response_json = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.model,
                            "choices": [choice],
                        }

                        if is_finished:
                            response_json["usage"] = {
                                "prompt_tokens": 0,
                                "completion_tokens": len(accumulated_text.split())
                                if accumulated_text
                                else 0,
                                "total_tokens": len(accumulated_text.split())
                                if accumulated_text
                                else 0,
                            }

                        if first_output_chunk_time is None:
                            first_output_chunk_time = time.perf_counter()
                        output_chunk_count += 1
                        yield response_json

                        if is_finished:
                            finished_sent = True
                            break

                except Exception as e:
                    logger.error(f"Error processing SGLang response: {e}")
                    yield {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": f"Error: {str(e)}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    if first_output_chunk_time is None:
                        first_output_chunk_time = time.perf_counter()
                    output_chunk_count += 1
                    break

            split_total_ms = (time.perf_counter() - split_start_time) * 1000.0
            split_first_upstream_chunk_ms = (
                (first_upstream_chunk_time - split_start_time) * 1000.0
                if first_upstream_chunk_time is not None
                else -1.0
            )
            split_first_output_chunk_ms = (
                (first_output_chunk_time - split_start_time) * 1000.0
                if first_output_chunk_time is not None
                else -1.0
            )
            logger.info(
                "split path metrics: request=%s first_upstream_chunk_ms=%.2f first_output_chunk_ms=%.2f total_ms=%.2f upstream_chunks=%d output_chunks=%d finished=%s",
                request_id,
                split_first_upstream_chunk_ms,
                split_first_output_chunk_ms,
                split_total_ms,
                upstream_chunk_count,
                output_chunk_count,
                finished_sent,
            )

            await readable.wait_for_completion()

    async def generate(self, raw_request: MultiModalRequest, context: Context):
        """
        Process multimodal request and forward to encode worker.

        Args:
            raw_request: Raw multimodal request to process.
            context: Context object for cancellation handling.
        """
        if not isinstance(raw_request, MultiModalRequest):
            # If the request is not MultiModalRequest, convert it to MultiModalRequest
            raw_request = MultiModalRequest.model_validate(raw_request)

        image_urls: list[str] = []
        video_url: str | None = None

        for message in raw_request.messages:
            for item in message.content:
                if item.type == "image_url":
                    if video_url is not None:
                        raise ValueError("Cannot provide both image and video URLs")
                    image_urls.append(item.image_url.url)
                elif item.type == "video_url":
                    if image_urls:
                        raise ValueError("Cannot provide both image and video URLs")
                    if video_url is not None:
                        raise ValueError("Multiple video URLs are not supported")
                    video_url = item.video_url.url

        if not image_urls and video_url is None:
            raise ValueError("Either image URL or video URL is required")

        multimodal_groups: list[MultiModalGroup] = []
        if image_urls:
            multimodal_groups = [
                MultiModalGroup(multimodal_input=MultiModalInput(image_url=url))
                for url in image_urls
            ]
        elif video_url is not None:
            multimodal_groups = [
                MultiModalGroup(multimodal_input=MultiModalInput(video_url=video_url))
            ]

        async for response in self._generate(raw_request, multimodal_groups):
            logger.debug(
                f"Generated response type {type(response)}, content: {response}"
            )
            yield response

    async def _generate(
        self,
        raw_request: MultiModalRequest,
        multimodal_groups: list[MultiModalGroup],
    ):
        # Generate a unique request ID for tracking
        request_id = str(uuid.uuid4().hex)
        logger.debug(f"Got raw request: {raw_request}")

        # Create SGLang conversation prompt
        sglang_request = multimodal_request_to_sglang(
            raw_request, self.tokenizer, self.chat_template
        )

        worker_request = SglangMultimodalRequest(
            request=sglang_request,
            multimodal_inputs=multimodal_groups,
        )

        valid_encode_instances = await self._count_valid_encode_instances()
        if self._can_split_encode(multimodal_groups, valid_encode_instances):
            await self._record_request_path(request_id, "split")
            try:
                async for item in self._generate_split(
                    request_id,
                    raw_request,
                    sglang_request,
                    multimodal_groups,
                ):
                    yield item
                return
            except Exception:
                logger.exception(
                    "split encode path failed for request %s; falling back to single-encoder path",
                    request_id,
                )

            await self._record_request_path(request_id, "single")

        # Send to encoder worker using load-aware routing.
        # Non-split policy:
        # - If any CPU instance is idle, route by global min in-flight across CPU/non-CPU.
        # - If CPU is busy, prefer non-CPU and avoid CPU when possible.
        single_path_start_time = time.perf_counter()
        has_idle_cpu = await self._has_idle_cpu_encode_instance()
        dispatch_start_time = time.perf_counter()
        if has_idle_cpu:
            response_generator, selected_instance = await self._dispatch_to_encoder(
                worker_request.model_dump_json()
            )
        else:
            response_generator, selected_instance = await self._dispatch_to_encoder(
                worker_request.model_dump_json(),
                prefer_device="none-cpu",
                avoid_device="cpu",
            )
        dispatch_latency_ms = (time.perf_counter() - dispatch_start_time) * 1000.0
        selected_device = (
            self._encoder_device.get(selected_instance, "unknown")
            if selected_instance is not None
            else "unknown"
        )

        # Process and yield SGLang responses
        finished_sent = False
        accumulated_text = ""
        first_upstream_chunk_time: float | None = None
        first_output_chunk_time: float | None = None
        upstream_chunk_count = 0
        output_chunk_count = 0

        try:
            async for resp in response_generator:
                upstream_chunk_count += 1
                if first_upstream_chunk_time is None:
                    first_upstream_chunk_time = time.perf_counter()
                try:
                    # Handle Annotated response objects from Dynamo (like vLLM pattern but for SGLang)
                    if hasattr(resp, "data"):
                        # Extract data from Dynamo Annotated response
                        raw_data = resp.data
                        if callable(raw_data):
                            raw_data = raw_data()

                        if isinstance(raw_data, str):
                            try:
                                response_data = json.loads(raw_data)
                            except json.JSONDecodeError:
                                response_data = {
                                    "text": raw_data,
                                    "finished": False,
                                }
                        else:
                            response_data = raw_data
                    elif isinstance(resp, str):
                        try:
                            response_data = json.loads(resp)
                        except json.JSONDecodeError:
                            response_data = {"text": resp, "finished": False}
                    else:
                        response_data = resp

                    # Use SGLang chat_processor for detokenization
                    (
                        text_content,
                        accumulated_text,
                        is_finished,
                    ) = process_sglang_stream_response(
                        response_data, self.tokenizer, accumulated_text
                    )

                    # Create OpenAI-compatible response (following vLLM-like pattern but for SGLang)
                    if text_content or is_finished:
                        choice: Dict[str, Any] = {
                            "index": 0,
                            "delta": {},
                            "finish_reason": None,
                        }
                        delta: Dict[str, str] = choice["delta"]  # Type-safe access

                        # Add role for first message or when there's content
                        if text_content and not finished_sent:
                            delta["role"] = "assistant"

                        # Add content if available
                        if text_content:
                            delta["content"] = text_content

                        # Set finish reason if completed
                        if is_finished:
                            choice["finish_reason"] = response_data.get(
                                "finish_reason", "stop"
                            )
                            if not finished_sent and not text_content:
                                # Final chunk needs role if it's the first chunk
                                delta["role"] = "assistant"

                        response_json = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.model,
                            "choices": [choice],
                        }

                        # Add usage only for final response
                        if is_finished:
                            response_json["usage"] = {
                                "prompt_tokens": 0,
                                "completion_tokens": len(accumulated_text.split())
                                if accumulated_text
                                else 0,
                                "total_tokens": len(accumulated_text.split())
                                if accumulated_text
                                else 0,
                            }

                        if first_output_chunk_time is None:
                            first_output_chunk_time = time.perf_counter()
                        output_chunk_count += 1
                        yield response_json

                        if is_finished:
                            finished_sent = True
                            break

                except Exception as e:
                    logger.error(f"Error processing SGLang response: {e}")
                    error_response = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": f"Error: {str(e)}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    if first_output_chunk_time is None:
                        first_output_chunk_time = time.perf_counter()
                    output_chunk_count += 1
                    yield error_response
                    break
        finally:
            total_latency_ms = (time.perf_counter() - single_path_start_time) * 1000.0
            first_upstream_chunk_ms = (
                (first_upstream_chunk_time - single_path_start_time) * 1000.0
                if first_upstream_chunk_time is not None
                else -1.0
            )
            first_output_chunk_ms = (
                (first_output_chunk_time - single_path_start_time) * 1000.0
                if first_output_chunk_time is not None
                else -1.0
            )
            logger.info(
                "single encode path metrics: request=%s selected_instance=%s selected_device=%s has_idle_cpu=%s dispatch_ms=%.2f first_upstream_chunk_ms=%.2f first_output_chunk_ms=%.2f total_ms=%.2f upstream_chunks=%d output_chunks=%d finished=%s",
                request_id,
                selected_instance,
                selected_device,
                has_idle_cpu,
                dispatch_latency_ms,
                first_upstream_chunk_ms,
                first_output_chunk_ms,
                total_latency_ms,
                upstream_chunk_count,
                output_chunk_count,
                finished_sent,
            )
            if selected_instance is not None:
                await self._on_encoder_request_done(selected_instance)

    async def _dispatch_to_encoder(
        self,
        payload: str,
        *,
        prefer_device: str | None = None,
        avoid_device: str | None = None,
    ):
        """Dispatch request to encoder worker using least in-flight routing.

        Returns:
            Tuple[AsyncIterator, Optional[int]]:
                - Response generator
                - Selected instance id if direct routing was used
        """
        instances = self.encode_worker_client.instance_ids()
        if not instances:
            instances = await self.encode_worker_client.wait_for_instances()

        await self._probe_encoder_devices(instances)

        async with self._encoder_route_lock:
            active_set = set(instances)
            for stale_instance in list(self._encoder_inflight.keys()):
                if stale_instance not in active_set:
                    del self._encoder_inflight[stale_instance]
            for stale_instance in list(self._encoder_device.keys()):
                if stale_instance not in active_set:
                    del self._encoder_device[stale_instance]

            for instance in instances:
                self._encoder_inflight.setdefault(instance, 0)

            sorted_instances = sorted(instances)
            if sorted_instances:
                start = self._encoder_rr_index % len(sorted_instances)
                tie_break_order = sorted_instances[start:] + sorted_instances[:start]

                candidate_instances = tie_break_order
                if avoid_device is not None:
                    filtered = [
                        instance
                        for instance in candidate_instances
                        if self._encoder_device.get(instance) != avoid_device
                    ]
                    if filtered:
                        candidate_instances = filtered

                if prefer_device is not None:
                    preferred = [
                        instance
                        for instance in candidate_instances
                        if self._encoder_device.get(instance) == prefer_device
                    ]
                    if preferred:
                        candidate_instances = preferred

                min_inflight = min(
                    self._encoder_inflight.get(instance, 0)
                    for instance in candidate_instances
                )
                min_inflight_instances = [
                    instance
                    for instance in candidate_instances
                    if self._encoder_inflight.get(instance, 0) == min_inflight
                ]
                selected_instance = next(
                    (
                        instance
                        for instance in min_inflight_instances
                        if self._encoder_device.get(instance) == "none-cpu"
                    ),
                    min_inflight_instances[0],
                )

                self._encoder_rr_index = (start + 1) % len(sorted_instances)
                self._encoder_inflight[selected_instance] += 1
            else:
                selected_instance = None

        if selected_instance is None:
            logger.info(
                "encoder dispatch route=round_robin selected_instance=None selected_device=unknown reason=no_active_instance"
            )
            return await self.encode_worker_client.round_robin(payload), None

        selected_device = self._encoder_device.get(selected_instance, "unknown")
        logger.info(
            "encoder dispatch route=direct selected_instance=%s selected_device=%s prefer_device=%s avoid_device=%s inflight=%s",
            selected_instance,
            selected_device,
            prefer_device,
            avoid_device,
            self._encoder_inflight.get(selected_instance, 0),
        )

        try:
            response_generator = await self.encode_worker_client.direct(
                payload, selected_instance
            )
            return response_generator, selected_instance
        except Exception:
            logger.exception(
                "Failed direct dispatch to encoder instance %s (device=%s), falling back to round_robin",
                selected_instance,
                selected_device,
            )
            await self._on_encoder_request_done(selected_instance)
            return await self.encode_worker_client.round_robin(payload), None

    async def _on_encoder_request_done(self, instance_id: int) -> None:
        """Mark completion of an encoder request for in-flight accounting."""
        async with self._encoder_route_lock:
            in_flight = self._encoder_inflight.get(instance_id, 0)
            if in_flight <= 1:
                self._encoder_inflight[instance_id] = 0
            else:
                self._encoder_inflight[instance_id] = in_flight - 1

    async def _probe_encoder_devices(self, instances: list[int]) -> None:
        unknown_instances = [
            instance for instance in instances if instance not in self._encoder_device
        ]
        if not unknown_instances:
            return

        async with self._encoder_probe_lock:
            probe_targets = [
                instance
                for instance in instances
                if instance not in self._encoder_device
            ]
            for instance in probe_targets:
                self._encoder_device[
                    instance
                ] = await self._probe_single_encoder_device(instance)

    async def _probe_single_encoder_device(self, instance: int) -> str:
        probe_payload = json.dumps({"_dynamo_probe_device": True})
        try:
            probe_stream = await self.encode_worker_client.direct(
                probe_payload, instance
            )
            first = await asyncio.wait_for(anext(probe_stream), timeout=2.0)
            probe_data = first.data() if hasattr(first, "data") else first

            if isinstance(probe_data, str):
                parsed = json.loads(probe_data)
            elif isinstance(probe_data, dict):
                parsed = probe_data
            else:
                parsed = {}

            device = parsed.get("device")
            if isinstance(device, str) and device in {"none-cpu", "cpu"}:
                return device
        except Exception:
            logger.debug(
                "Failed probing device for encoder instance %s; treating as unknown",
                instance,
                exc_info=True,
            )
        return "unknown"
