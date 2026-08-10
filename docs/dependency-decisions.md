# Dependency Decisions

Status: Phase 0 decision record (2026-08-09)

## Decision Summary

| Capability | MVP decision | Optional integration | Reason |
| --- | --- | --- | --- |
| Python | 3.11 target | 3.10 supported | Common supported range for LingBot-Map and Habitat-Lab |
| Core models | Typed dataclasses plus NumPy | Pydantic at config/API edges | Keep the mock path small and offline |
| Configuration | YAML plus environment and CLI overrides | OmegaConf later if needed | No dependency on Habitat's Hydra tree in core code |
| Logging | Standard library JSONL formatter | structlog adapter | Reliable fallback with no service dependency |
| Scene graph | NetworkX behind a graph protocol | Custom indexed graph later | Mature directed multigraph and serialization support |
| Structured memory | SQLite | PostgreSQL | Built in, transactional, portable |
| Vector retrieval | SQLite metadata plus NumPy cosine search | FAISS/Chroma/Qdrant | Tests must not require native or remote services |
| Mapping | Deterministic `MockMapper` | LingBot-Map | GPU/checkpoint independent baseline |
| Perception | Deterministic mock | HF VLM or Grounding-DINO plus SAM | Replaceable and training-free |
| Planning | Deterministic rule planner | OpenAI-compatible or local HF LLM | Reproducible fallback and valid structured output |
| Execution | Unitree-like planar simulator | Habitat, ROS2/Gazebo/Isaac/Unitree | Prevent simulator/hardware coupling |
| Evaluation | NumPy implementations | evo/Open3D accelerators | Lightweight smoke tests first |

## External Repository Decisions

### LingBot-Map

Decision: install as an optional external package or editable checkout. Do not copy its
implementation into the new package.

Verified interface:

- Package requires Python 3.10 or newer in metadata; README recommends Python 3.10.
- README recommends Torch 2.8.0 with CUDA 12.8.
- FlashInfer is the optimized streaming backend; PyTorch SDPA is the fallback.
- `GCTStream.inference_streaming` accepts `[S,3,H,W]` or `[B,S,3,H,W]` images in
  `[0,1]` and returns `pose_enc`, `depth`, `depth_conf`, `world_points`, and
  `world_points_conf` when the corresponding heads are enabled.
- The method performs scale-frame initialization and one-frame causal updates using a
  KV cache. `clean_kv_cache()` resets sequence state.
- Windowed inference is available for long offline sequences.

Adapter decision:

- Implement online `update(frame)` by reproducing only the public demo call sequence:
  initial scale-frame `forward`, followed by one-frame `forward` calls, including the
  keyframe `_set_skip_append` behavior where supported.
- Keep a compatibility shim because `_set_skip_append` is effectively internal.
- Fall back to bounded sequence calls if an upstream version changes internals.
- Save decoded products and keyframes; reconstruct ephemeral KV state by replay.

License: local source is Apache-2.0. No `NOTICE` file is present. The checkpoint/model
card license was not verified from local files and must be checked before redistribution
or publication.

Open issues:

- Demo code decodes the pose then inverts it as if it were world-to-camera; benchmark
  code comments that the same decoder returns camera-to-world. A synthetic or known-pose
  fixture must resolve this before integration is accepted.
- CPU execution is coded but expected to be slow and has not been tested here.
- `world_points` availability depends on model/head configuration; depth back-projection
  is the adapter fallback.

### Habitat-Lab and Habitat-Sim

Decision: optional simulation extra, imported only inside `HabitatAdapter`.

Verified capabilities:

- Habitat-Lab 0.3.3 metadata supports Python 3.9 through 3.11 and pins NumPy 1.26.4.
- Habitat-Sim provides RGB-D sensors, egomotion/agent state, NavMesh pathfinding,
  navigability checks, semantic scene metadata, and previous-step collision state.
- Habitat navigation has discrete actions; rearrangement environments support
  continuous base velocity actions.
- Bullet-enabled builds are required for physics-backed articulated simulation.
- Source is MIT licensed.

Boundary decision:

- The core executor works against `RobotBackend`, not Habitat classes.
- Habitat provides observations, navigability, path planning, and collision feedback.
- The Unitree-like controller converts waypoints into bounded planar base commands.
- Habitat Spot/AlienGo examples do not establish Unitree Go2 controller fidelity.
- A Go2 URDF, actuator model, contact behavior, and controller require a separate,
  explicitly enabled backend and dedicated validation.

Installation decision: use a separate Python 3.11 environment and a matched stable
Habitat-Lab/Habitat-Sim pair. Prefer a headless, Bullet-enabled build on this machine.
Do not install Habitat into the current Python 3.12 base interpreter.

### Open3DSG

Decision: preserve its graph semantics and open-vocabulary query concept, but do not
make Open3DSG a core runtime dependency.

Reasons:

- Open3DSG preprocessing assumes presegmented object instances and bounded offline
  subgraphs.
- Its SGPN predicts node and relationship embeddings using PointNet, GNN, CLIP/OpenSeg,
  and optional BLIP/LLaVA distillation. The requested system must be training-free.
- The source is AGPL-3.0, which creates distribution and network-use obligations that
  should not silently propagate into a differently licensed core project.
- Its package metadata does not declare the heavy runtime dependency set.

Integration approach:

- Retain directed subject-predicate-object edges and object-centric nodes.
- Extend node types to room, region, robot, obstacle, and traversable region.
- Use geometric and VLM inference at runtime instead of SGPN training.
- Provide an optional out-of-process Open3DSG feature importer later if compatibility
  with existing checkpoints or exported graphs is required.
- Do not copy Open3DSG implementation code into the new package without an explicit
  project licensing decision.

## Environment Inspection

Observed on 2026-08-09:

```text
OS: Linux
Python: 3.12.3 at /usr/local/bin/python
torch: not installed
habitat: not installed
habitat_sim: not installed
lingbot_map: not installed
NVIDIA driver query: failed; nvidia-smi could not communicate with the driver
```

Repository state: the root repository currently reports the existing project files and
cloned dependencies as untracked. Phase work must not delete, reset, or rewrite those
files. The new package should coexist with the checked-out external repositories.

## Proposed Environment Layout

Use one lightweight development environment first:

```text
agentic-memory-nav-core (Python 3.11)
  numpy, networkx, pyyaml, pytest, pytest-asyncio, ruff, mypy
```

Heavy extras are isolated by dependency group and lazy imports:

```text
mapping-lingbot: torch/torchvision, lingbot-map, optional flashinfer
simulation-habitat: matched habitat-lab and habitat-sim with Bullet
perception-hf: torch, transformers, accelerate
perception-grounding: detector and segmenter packages
planning-openai: OpenAI-compatible client only
vector-faiss: faiss-cpu or faiss-gpu
```

If LingBot-Map and Habitat dependency constraints conflict, run them as separate worker
processes with versioned JSON/NPZ artifact contracts rather than forcing one environment.

## Dataset Decisions

- ETH3D is a geometry and trajectory evaluation source, not a complete VLN benchmark.
- Habitat scenes provide simulation observations and navigation tasks; licenses and
  asset downloads remain external.
- RGB-D and custom videos need sidecar calibration and optional pose files. Missing
  calibration is explicit and cannot be silently replaced with arbitrary intrinsics.
- LingBot-Map examples are geometry smoke-test inputs, subject to their own data terms.
- Language instructions and reference trajectories use a separate adapter so they are
  not incorrectly inferred from ETH3D metadata.

## Assumptions Requiring Validation

1. A Python 3.11 environment can host the core package and at least one matched Habitat
   release without local compilation failures.
2. The target machine will eventually expose a usable NVIDIA driver and sufficient GPU
   memory for the chosen LingBot-Map checkpoint.
3. LingBot-Map weights permit the intended research use and artifact distribution.
4. LingBot-Map pose decoding can be normalized to camera-to-world with a known-pose test.
5. Replaying retained LingBot keyframes reproduces acceptable online state after load.
6. Habitat scene assets include the semantic metadata needed by selected evaluations.
7. Planar Unitree-like kinematics are sufficient for the first navigation experiments;
   they do not represent real Go2 gait or contact dynamics.
8. A training-free detector/VLM can provide stable categories and appearance embeddings
   at the required latency; mock behavior is not evidence of real-model performance.
9. SQLite plus local cosine search is sufficient for MVP memory scale.
10. Coordinate and timestamp sources from robot streams are synchronized or expose
    enough metadata to estimate alignment uncertainty.

## Phase 0 Exit Criteria

- Architecture, ownership boundaries, contracts, safety invariants, and persistence
  strategy are documented.
- External APIs and licenses are recorded only where verified locally.
- Unverified model, pose, hardware, and dataset assumptions are explicit.
- No third-party API has been fabricated and no heavy dependency has been installed.
- Phase 1 may begin with a Python 3.11, mock-first package skeleton.