"""Typed domain contracts shared by all agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

Vector3 = tuple[float, float, float]
BBox2D = tuple[int, int, int, int]
BBox3D = tuple[float, float, float, float, float, float]
FloatArray = NDArray[np.floating[Any]]
UInt8Array = NDArray[np.uint8]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class NodeType(StrEnum):
    OBJECT = "object"
    ROOM = "room"
    REGION = "region"
    ROBOT = "robot"
    OBSTACLE = "obstacle"
    TRAVERSABLE = "traversable"


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    TASK = "task"
    UNCERTAINTY = "uncertainty"


class TaskStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ActionType(StrEnum):
    STOP = "stop"
    EXPLORE = "explore"
    NAVIGATE = "navigate"
    VERIFY = "verify"


@dataclass(slots=True)
class Pose3D:
    position: Vector3 = (0.0, 0.0, 0.0)
    yaw: float = 0.0


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(slots=True)
class FrameObservation:
    frame_id: str
    timestamp: float
    rgb: UInt8Array
    depth: FloatArray | None = None
    camera_intrinsics: CameraIntrinsics | None = None
    camera_pose: Pose3D | None = None
    robot_pose: Pose3D | None = None
    source: str = "simulated"
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MappingUpdate:
    frame_id: str
    timestamp: float
    camera_pose: Pose3D
    depth: FloatArray
    confidence: FloatArray
    local_pointcloud: FloatArray
    global_pointcloud: FloatArray
    is_keyframe: bool
    map_version: int
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InstanceGeometry:
    """Compact reference to an instance point cloud stored outside graph JSON."""

    instance_id: str
    artifact_path: str | None
    point_count: int
    centroid_3d: Vector3
    dimensions_3d: Vector3
    coordinate_frame: str
    confidence: float
    mask_provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RelationEvidence:
    """A time-stamped observation supporting a directed graph predicate."""

    source: str
    predicate: str
    confidence: float
    timestamp: float
    source_frame: str
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SceneTriple:
    """A semantic subject-predicate-object claim between frame observations."""

    subject_observation_id: str
    predicate: str
    object_observation_id: str
    confidence: float
    timestamp: float
    frame_id: str
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ObjectObservation:
    observation_id: str
    category: str
    attributes: dict[str, str]
    bbox_2d: BBox2D
    center_3d: Vector3
    dimensions_3d: Vector3
    confidence: float
    timestamp: float
    frame_id: str
    embedding: FloatArray | None = None
    track_id: str | None = None
    geometry: InstanceGeometry | None = None
    provenance: list[str] = field(default_factory=list)

    @property
    def bbox_3d(self) -> BBox3D:
        half = tuple(value / 2.0 for value in self.dimensions_3d)
        return (
            self.center_3d[0] - half[0],
            self.center_3d[1] - half[1],
            self.center_3d[2] - half[2],
            self.center_3d[0] + half[0],
            self.center_3d[1] + half[1],
            self.center_3d[2] + half[2],
        )


@dataclass(slots=True)
class SceneNode:
    node_id: str
    node_type: NodeType
    label: str
    attributes: dict[str, str]
    position_3d: Vector3
    bbox_3d: BBox3D | None
    uncertainty: float
    first_seen: float
    last_seen: float
    confidence: float
    source_frame: str
    observation_count: int = 1
    observation_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    geometry: InstanceGeometry | None = None
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SceneEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    confidence: float
    first_seen: float
    last_seen: float
    source_frame: str
    uncertainty: float = 0.0
    observation_count: int = 1
    position_3d: Vector3 | None = None
    evidence: list[RelationEvidence] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryItem:
    memory_id: str
    memory_type: MemoryType
    content: str
    structured_payload: dict[str, Any]
    timestamp: float
    location: Vector3 | None
    confidence: float
    provenance: list[str]
    embedding: list[float] | None = None
    decay_score: float = 1.0
    version: int = 1


@dataclass(slots=True)
class NavigationTask:
    task_id: str
    natural_language_instruction: str
    parsed_goal: dict[str, Any]
    subgoals: list[dict[str, Any]]
    constraints: list[str]
    status: TaskStatus = TaskStatus.PENDING
    current_subgoal: int = 0


@dataclass(slots=True)
class ActionIntent:
    action_id: str
    action_type: ActionType
    target: str | None
    waypoint: Vector3 | None
    duration: float
    safety_constraints: list[str]
    confidence: float
    reason: str
    expected_observation: str | None = None


@dataclass(slots=True)
class ExecutionFeedback:
    action_id: str
    success: bool
    state: Pose3D
    collision: bool
    elapsed: float
    reason: str


@dataclass(slots=True)
class NavigationPlan:
    task_id: str
    goal: str
    subgoals: list[dict[str, Any]]
    required_entities: list[str]
    waypoints: list[Vector3]
    information_gaps: list[str]
    assumptions: list[str]
    confidence: float
    replan_required: bool
    action: ActionIntent


def jsonable(value: Any) -> Any:
    """Convert domain values and arrays into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value
