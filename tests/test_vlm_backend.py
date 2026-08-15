import json

import numpy as np

from agentic_memory_nav.common.types import (
    CameraIntrinsics,
    FrameObservation,
    MappingUpdate,
    Pose3D,
)
from agentic_memory_nav.perception.vlm_backend import VLMBackend


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_vlm_backend_parses_object_list(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "objects": [
                                {
                                    "category": "cup",
                                    "attributes": {"color": "red"},
                                    "bbox_2d": [10, 20, 40, 60],
                                    "center_3d": [2.0, 0.8, 2.2],
                                    "dimensions_3d": [0.12, 0.16, 0.12],
                                    "confidence": 0.96,
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }

    def fake_urlopen(request, timeout=None):
        return DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = VLMBackend(
        model_id="demo-model",
        api_key="demo-key",
        base_url="http://example.com/v1",
        api="openai-completions",
    )

    frame = FrameObservation(
        frame_id="frame_0",
        timestamp=0.0,
        rgb=np.zeros((64, 96, 3), dtype=np.uint8),
        depth=np.full((64, 96), 2.0, dtype=np.float32),
        camera_intrinsics=CameraIntrinsics(80.0, 80.0, 48.0, 32.0, 96, 64),
        camera_pose=Pose3D(position=(0.0, 0.0, 0.0), yaw=0.0),
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0), yaw=0.0),
    )
    mapping = MappingUpdate(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        camera_pose=frame.camera_pose,
        depth=frame.depth,
        confidence=np.ones_like(frame.depth, dtype=np.float32),
        local_pointcloud=np.zeros((1, 3), dtype=np.float32),
        global_pointcloud=np.zeros((1, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
        provenance=["test"],
    )

    objects = backend.detect(frame, mapping)
    assert len(objects) == 1
    assert objects[0].category == "cup"
    assert objects[0].attributes["color"] == "red"


def test_vlm_backend_sends_thinking_switch_at_top_level(monkeypatch):
    """vLLM ignores a nested `extra_body`, which silently re-enables reasoning output."""
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        captured.update(json.loads(request.data.decode("utf-8")))
        return DummyResponse({"choices": [{"message": {"content": '{"objects": []}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = VLMBackend(
        model_id="demo-model",
        base_url="http://example.com/v1",
        api="openai-completions",
    )
    frame = FrameObservation(
        frame_id="frame_0",
        timestamp=0.0,
        rgb=np.zeros((64, 96, 3), dtype=np.uint8),
        depth=np.full((64, 96), 2.0, dtype=np.float32),
        camera_intrinsics=CameraIntrinsics(80.0, 80.0, 48.0, 32.0, 96, 64),
        camera_pose=Pose3D(position=(0.0, 0.0, 0.0), yaw=0.0),
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0), yaw=0.0),
    )
    mapping = MappingUpdate(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        camera_pose=frame.camera_pose,
        depth=frame.depth,
        confidence=np.ones_like(frame.depth, dtype=np.float32),
        local_pointcloud=np.zeros((1, 3), dtype=np.float32),
        global_pointcloud=np.zeros((1, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
        provenance=["test"],
    )

    backend.detect(frame, mapping)

    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in captured


def test_vlm_backend_extracts_json_from_reasoning_response(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "reasoning": (
                        "I will answer carefully. ```json "
                        '{"objects": [{"category": "cup", "attributes": '
                        '{"color": "red"}, "bbox_2d": [10, 20, 40, 60], '
                        '"center_3d": [2.0, 0.8, 2.2], "dimensions_3d": '
                        '[0.12, 0.16, 0.12], "confidence": 0.96}]} ```'
                    ),
                    "content": None,
                }
            }
        ]
    }

    def fake_urlopen(request, timeout=None):
        return DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    backend = VLMBackend(
        model_id="demo-model",
        api_key="demo-key",
        base_url="http://example.com/v1",
        api="openai-completions",
    )

    frame = FrameObservation(
        frame_id="frame_1",
        timestamp=1.0,
        rgb=np.zeros((64, 96, 3), dtype=np.uint8),
        depth=np.full((64, 96), 2.0, dtype=np.float32),
        camera_intrinsics=CameraIntrinsics(80.0, 80.0, 48.0, 32.0, 96, 64),
        camera_pose=Pose3D(position=(0.0, 0.0, 0.0), yaw=0.0),
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0), yaw=0.0),
    )
    mapping = MappingUpdate(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        camera_pose=frame.camera_pose,
        depth=frame.depth,
        confidence=np.ones_like(frame.depth, dtype=np.float32),
        local_pointcloud=np.zeros((1, 3), dtype=np.float32),
        global_pointcloud=np.zeros((1, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
        provenance=["test"],
    )

    objects = backend.detect(frame, mapping)
    assert len(objects) == 1
    assert objects[0].category == "cup"
    assert objects[0].attributes["color"] == "red"


def test_vlm_backend_sends_rgb_and_maps_indexed_triples(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "objects": [
                                {
                                    "category": "cup",
                                    "attributes": {"color": "red"},
                                    "bbox_2d": [10, 20, 40, 60],
                                    "center_3d": [2.0, 0.8, 2.2],
                                    "dimensions_3d": [0.12, 0.16, 0.12],
                                    "confidence": 0.96,
                                },
                                {
                                    "category": "table",
                                    "attributes": {},
                                    "bbox_2d": [0, 0, 95, 63],
                                    "center_3d": [2.0, 0.0, 2.2],
                                    "dimensions_3d": [1.0, 0.7, 1.0],
                                    "confidence": 0.9,
                                },
                            ],
                            "triples": [
                                {
                                    "subject_index": 0,
                                    "predicate": "on_top_of",
                                    "object_index": 1,
                                    "confidence": 0.88,
                                }
                            ],
                        }
                    )
                }
            }
        ]
    }
    sent_payload = {}

    def fake_urlopen(http_request, timeout=None):
        sent_payload.update(json.loads(http_request.data.decode("utf-8")))
        return DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = VLMBackend(model_id="demo-model", base_url="http://example.com/v1")
    frame = FrameObservation(
        frame_id="frame_rgb",
        timestamp=1.0,
        rgb=np.full((2, 3, 3), 255, dtype=np.uint8),
        depth=np.ones((2, 3), dtype=np.float32),
        camera_intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 3, 2),
        camera_pose=Pose3D(),
    )
    mapping = MappingUpdate(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        camera_pose=Pose3D(),
        depth=frame.depth,
        confidence=np.ones_like(frame.depth),
        local_pointcloud=np.zeros((1, 3), dtype=np.float32),
        global_pointcloud=np.zeros((1, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
    )

    observations, triples = backend.analyze(frame, mapping)

    image = sent_payload["messages"][0]["content"][-1]["image_url"]["url"]
    assert image.startswith("data:image/png;base64,")
    assert len(observations) == 2
    assert triples[0].subject_observation_id == observations[0].observation_id
    assert triples[0].object_observation_id == observations[1].observation_id
    assert triples[0].predicate == "on_top_of"


def test_vlm_backend_normalizes_floating_point_bounding_boxes(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "objects": [
                                {
                                    "category": "box",
                                    "attributes": {},
                                    "bbox_2d": [1.2, 2.7, 10.1, 20.9],
                                    "center_3d": [0.0, 0.0, 1.0],
                                    "dimensions_3d": [1.0, 1.0, 1.0],
                                    "confidence": 0.9,
                                }
                            ],
                            "triples": [],
                        }
                    )
                }
            }
        ]
    }

    def fake_urlopen(request, timeout=None):
        return DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = VLMBackend(model_id="demo-model", base_url="http://example.com/v1")
    frame = FrameObservation("frame_2", 2.0, np.zeros((32, 32, 3), dtype=np.uint8))
    mapping = MappingUpdate(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        camera_pose=Pose3D(),
        depth=np.ones((32, 32), dtype=np.float32),
        confidence=np.ones((32, 32), dtype=np.float32),
        local_pointcloud=np.zeros((1, 3), dtype=np.float32),
        global_pointcloud=np.zeros((1, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
    )

    observations, _ = backend.analyze(frame, mapping)

    assert observations[0].bbox_2d == (1, 3, 10, 21)


def test_vlm_backend_scales_qwen_style_normalized_bounding_boxes(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "objects": [
                                {
                                    "category": "target",
                                    "attributes": {},
                                    "bbox_2d": [100, 100, 900, 800],
                                    "center_3d": [0.0, 0.0, 1.0],
                                    "dimensions_3d": [1.0, 1.0, 1.0],
                                    "confidence": 0.9,
                                }
                            ],
                            "triples": [],
                        }
                    )
                }
            }
        ]
    }

    def fake_urlopen(request, timeout=None):
        return DummyResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    backend = VLMBackend(model_id="demo-model", base_url="http://example.com/v1")
    frame = FrameObservation("frame_3", 3.0, np.zeros((192, 256, 3), dtype=np.uint8))
    mapping = MappingUpdate(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        camera_pose=Pose3D(),
        depth=np.ones((192, 256), dtype=np.float32),
        confidence=np.ones((192, 256), dtype=np.float32),
        local_pointcloud=np.zeros((1, 3), dtype=np.float32),
        global_pointcloud=np.zeros((1, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
    )

    observations, _ = backend.analyze(frame, mapping)

    assert observations[0].bbox_2d == (26, 19, 230, 154)
