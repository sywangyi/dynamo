# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, Optional

from transformers import AutoTokenizer

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
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(config)
        self.encode_worker_client = encode_worker_client
        self.chat_template = getattr(config.server_args, "chat_template", "qwen2-vl")
        self.model = config.server_args.model_path
        self.shutdown_event = shutdown_event
        self._encoder_inflight: dict[int, int] = defaultdict(int)
        self._encoder_device: dict[int, str] = {}
        self._encoder_route_lock = asyncio.Lock()
        self._encoder_probe_lock = asyncio.Lock()
        self._encoder_rr_index = 0

        # Initialize tokenizer for the model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
            truncation_side="left",
        )

    def cleanup(self):
        pass

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

        # Send to encoder worker using load-aware routing
        response_generator, selected_instance = await self._dispatch_to_encoder(
            worker_request.model_dump_json()
        )

        # Process and yield SGLang responses
        finished_sent = False
        accumulated_text = ""

        try:
            async for resp in response_generator:
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
                    yield error_response
                    break
        finally:
            if selected_instance is not None:
                await self._on_encoder_request_done(selected_instance)

    async def _dispatch_to_encoder(self, payload: str):
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
                cpu_busy = any(
                    self._encoder_device.get(instance) == "cpu"
                    and self._encoder_inflight.get(instance, 0) > 0
                    for instance in tie_break_order
                )

                if cpu_busy:
                    non_cpu_instances = [
                        instance
                        for instance in tie_break_order
                        if self._encoder_device.get(instance) == "none-cpu"
                    ]
                    if non_cpu_instances:
                        min_inflight = min(
                            self._encoder_inflight.get(instance, 0)
                            for instance in non_cpu_instances
                        )
                        selected_instance = next(
                            instance
                            for instance in non_cpu_instances
                            if self._encoder_inflight.get(instance, 0)
                            == min_inflight
                        )
                    else:
                        min_inflight = min(
                            self._encoder_inflight.get(instance, 0)
                            for instance in tie_break_order
                        )
                        selected_instance = next(
                            instance
                            for instance in tie_break_order
                            if self._encoder_inflight.get(instance, 0)
                            == min_inflight
                        )
                else:
                    min_inflight = min(
                        self._encoder_inflight.get(instance, 0)
                        for instance in tie_break_order
                    )
                    min_inflight_candidates = [
                        instance
                        for instance in tie_break_order
                        if self._encoder_inflight.get(instance, 0) == min_inflight
                    ]
                    non_cpu_tied_candidates = [
                        instance
                        for instance in min_inflight_candidates
                        if self._encoder_device.get(instance) == "none-cpu"
                    ]
                    selected_instance = (
                        non_cpu_tied_candidates[0]
                        if non_cpu_tied_candidates
                        else min_inflight_candidates[0]
                    )

                self._encoder_rr_index = (start + 1) % len(sorted_instances)
                self._encoder_inflight[selected_instance] += 1
            else:
                selected_instance = None

        if selected_instance is None:
            return await self.encode_worker_client.round_robin(payload), None

        try:
            response_generator = await self.encode_worker_client.direct(
                payload, selected_instance
            )
            return response_generator, selected_instance
        except Exception:
            logger.exception(
                "Failed direct dispatch to encoder instance %s, falling back to round_robin",
                selected_instance,
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
