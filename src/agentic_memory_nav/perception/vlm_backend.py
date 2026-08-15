"""Minimal OpenAI-compatible VLM perception backend.

This backend intentionally keeps the runtime surface small and deterministic:
- it issues a single chat/completions request;
- it expects a JSON payload with an `objects` list;
- it converts that list into the project’s common `ObjectObservation` format.

The implementation is intentionally lightweight so the system can be tested without
requiring a full local model stack or a full multimodal VLM runtime.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any
from urllib import request

import numpy as np
from PIL import Image

from agentic_memory_nav.common.types import (
    FrameObservation,
    MappingUpdate,
    ObjectObservation,
    SceneTriple,
    new_id,
)


class VLMBackend:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api: str = "openai-completions",
        timeout: float = 30.0,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.api_key = api_key or ""
        self.base_url = base_url or "http://localhost:8000/v1"
        self.api = api
        self.timeout = timeout

    def detect(self, frame: FrameObservation, mapping: MappingUpdate) -> list[ObjectObservation]:
        observations, _ = self.analyze(frame, mapping)
        return observations

    def analyze(
        self, frame: FrameObservation, mapping: MappingUpdate
    ) -> tuple[list[ObjectObservation], list[SceneTriple]]:
        payload = self._build_request(frame, mapping)
        result = self._post_json(payload)
        response = self._parse_payload(result)
        objects = response.get("objects", [])
        observations = [
            ObjectObservation(
                observation_id=new_id("obs"),
                category=item["category"],
                attributes=item.get("attributes", {}),
                bbox_2d=self._bbox_2d(
                    item.get("bbox_2d", (0, 0, 0, 0)),
                    width=frame.rgb.shape[1],
                    height=frame.rgb.shape[0],
                ),
                center_3d=tuple(item.get("center_3d", (0.0, 0.0, 0.0))),
                dimensions_3d=tuple(item.get("dimensions_3d", (0.0, 0.0, 0.0))),
                confidence=float(item.get("confidence", 0.0)),
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                embedding=np.array(item.get("embedding", [0.0, 0.0, 0.0]), dtype=np.float32),
                provenance=[frame.frame_id, self.model_id, "vlm_backend"],
            )
            for item in objects
        ]
        triples = self._parse_triples(response.get("triples", []), observations, frame)
        return observations, triples

    @staticmethod
    def _bbox_2d(value: object, *, width: int, height: int) -> tuple[int, int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return (0, 0, 0, 0)
        try:
            left, top, right, bottom = (float(component) for component in value)
        except (TypeError, ValueError):
            return (0, 0, 0, 0)
        # Qwen-compatible VLMs commonly emit image coordinates normalized to 0..1000.
        if max(left, top, right, bottom) > max(width, height) and all(
            0.0 <= component <= 1000.0 for component in (left, top, right, bottom)
        ):
            left, right = left * width / 1000.0, right * width / 1000.0
            top, bottom = top * height / 1000.0, bottom * height / 1000.0
        return (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))

    def _build_request(self, frame: FrameObservation, mapping: MappingUpdate) -> dict[str, Any]:
        prompt = (
            "You are a robot perception system. "
            "Return only valid JSON with `objects` and `triples` lists. "
            "Do not include markdown fences, reasoning, or explanatory text. "
            "Each object must contain: category, attributes, bbox_2d, center_3d, "
            "dimensions_3d, confidence. "
            "Each triple must contain: subject_index, predicate, object_index, confidence. "
            "Indices refer to positions in `objects`. If uncertain, return empty lists."
        )
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": f"Frame id: {frame.frame_id}"},
                        {
                            "type": "text",
                            "text": (
                                "Current map statistics: "
                                f"map_version={mapping.map_version}, "
                                f"points={mapping.global_pointcloud.shape[0]}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._rgb_data_url(frame.rgb)},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        if self.api == "openai-completions":
            # `extra_body` is an OpenAI *client* construct that the client flattens into
            # the request body. This backend posts raw HTTP, so the field must be sent at
            # the top level or vLLM silently ignores it and keeps emitting reasoning text.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    def _parse_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        try:
            choice = result["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                return {}

            candidates: list[str] = []
            if isinstance(message.get("content"), str):
                candidates.append(message["content"])
            elif isinstance(message.get("content"), list):
                candidates.append(
                    "".join(
                        str(part.get("text", ""))
                        for part in message["content"]
                        if isinstance(part, dict)
                    )
                )
            if isinstance(message.get("reasoning"), str):
                candidates.append(message["reasoning"])

            payload = None
            for candidate in candidates:
                parsed = self._extract_json_payload(candidate)
                if parsed is not None:
                    payload = parsed
                    break

            if not isinstance(payload, dict):
                return {}
            raw = payload.get("objects", [])
            payload["objects"] = [item for item in raw if self._valid_object(item)]
            triples = payload.get("triples", [])
            payload["triples"] = [item for item in triples if isinstance(item, dict)]
            return payload
        except (KeyError, IndexError, TypeError, ValueError):
            return {}

    @staticmethod
    def _valid_object(item: object) -> bool:
        return (
            isinstance(item, dict)
            and isinstance(item.get("category"), str)
            and isinstance(item.get("attributes", {}), dict)
        )

    def _parse_triples(
        self,
        raw_triples: object,
        observations: list[ObjectObservation],
        frame: FrameObservation,
    ) -> list[SceneTriple]:
        if not isinstance(raw_triples, list):
            return []
        triples: list[SceneTriple] = []
        for item in raw_triples:
            if not isinstance(item, dict):
                continue
            subject_index = item.get("subject_index")
            object_index = item.get("object_index")
            predicate = item.get("predicate")
            if (
                not isinstance(subject_index, int)
                or not isinstance(object_index, int)
                or not isinstance(predicate, str)
                or not predicate
                or subject_index < 0
                or object_index < 0
                or subject_index >= len(observations)
                or object_index >= len(observations)
            ):
                continue
            triples.append(
                SceneTriple(
                    subject_observation_id=observations[subject_index].observation_id,
                    predicate=predicate,
                    object_observation_id=observations[object_index].observation_id,
                    confidence=float(item.get("confidence", 0.0)),
                    timestamp=frame.timestamp,
                    frame_id=frame.frame_id,
                    provenance=[frame.frame_id, self.model_id, "vlm_triple"],
                )
            )
        return triples

    @staticmethod
    def _rgb_data_url(rgb: np.ndarray) -> str:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _extract_json_payload(content: str) -> dict[str, Any] | list[Any] | None:
        text = content.strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
