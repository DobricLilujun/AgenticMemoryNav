# AgenticMemoryNav — Coordinate-System & Camera Reference (STANDARDIZED)

> Read this before touching `isaacsim_adapter.py`. This document describes the
> **standardized** coordinate convention. The old `(x, height, north)` / north↔height
> swap and the `+90°` forward hack have been **removed**.

## 1. The coordinate frames (standardized)

| Frame | Axes | Used by |
|-------|------|---------|
| **odom (world)** | **X, Y, Z-up** | Isaac Sim USD world, `_robot`, cameras, PhysX. **`Pose3D` = odom exactly (no remap).** |
| **base_link** | **X forward, Y left, Z up** (ROS REP 105) | The robot body frame. Heading `yaw` about odom Z; `yaw=0` ⇒ base +X ∥ odom +X. |
| **camera_link** | offset in base_link frame `(forward, left, up)` | The head camera mount on the robot; moves with base_link. |
| **camera optical** | **−Z forward, +Y up, +X right** (Isaac default) | Isaac `Camera` primitive intrinsic frame. |

### Key change vs. the old code
- **No north↔height swap.** `_pose_to_usd` / `_usd_to_pose` are now **identity**:
  `Pose3D(x, y, z)` ↔ USD `(x, y, z)`, Z-up. (Previously height lived at index 1.)
- **No `+90°` forward hack.** The camera forward is simply the yaw-rotated base
  `+X`; `_GO2_FORWARD_YAW_RAD` is removed.
- **`base_link` is X-forward / Y-left / Z-up** (ROS REP 105), not the old
  "Go2 front = +Y" convention.

### Scene up-axis wrap (Y-up InternScenes → Z-up stage)
InternScenes scenes are **Y-up**; Isaac renders **Z-up**. `_open_scene_stage`
wraps the scene in a temp USD layer that rotates the scene onto Z-up:
```
/World (upAxis = Z, identity transform)
  /World/scene (prepend references = @scene@, xformOp:orient = (0.707,0.707,0,0))
```
Only the scene mesh is rotated; the robot and cameras live in **odom (frame A, Z-up)**.

## 2. Robot / yaw
- `self._yaw` is a **body yaw about odom Z**. `yaw=0` ⇒ base_link +X ∥ odom +X.
- base_link axes in odom: `+X → (cos, sin, 0)`, `+Y(left) → (−sin, cos, 0)`, `+Z(up) → (0,0,1)`.
- `_yaw_to_quat_wxyz(yaw)` sets the robot prim orientation (about odom Z).

## 3. Head camera (agent lens) — parented to the robot
The lens is a child prim of the robot Xform; its local pose (translation +
base orientation) is set once at construction time and the world pose is
carried automatically by the USD parent/child hierarchy. Only an armed scan
(`trigger_head_scan`) perturbs the local orientation with a small yaw/pitch;
otherwise the lens holds still and points straight ahead with the body.
- `_GO2_CAMERA_OFFSET_M = (-0.20, 0.0, 0.14)` = **(forward, left, up)** in base_link metres.
- **Static by default.** Only an armed scan (`trigger_head_scan`) adds a yaw/pitch;
  otherwise the lens holds and points straight ahead.

## 4. Optical → base_link alignment (documented, not applied by default)
The head camera is **world-placed**, so a look-at whose `−Z` points along base
`+X` already makes the camera look forward. No extra alignment is applied.
The constant `_OPTICAL_TO_BASE_OFFSET_WXYZ = [0.5, 0.5, −0.5, −0.5]` records the
fixed rotation that would map Isaac optical axes onto base_link axes
(optical +X→base −Y, +Y→+Z, +Z→−X); it is only used if a caller explicitly passes
an `orientation_offset` (e.g. for a **parented** camera chain).

## 5. Camera intrinsics
`get_observation` currently reports `CameraIntrinsics(80, 80, cx, cy, w, h)` and
does not yet derive the reported intrinsics from `camera_focal_length`. The config
`camera_focal_length` still drives the **rendered** FoV via Isaac `set_focal_length`;
the intrinsics reported to the mapping agent stay a fixed approximation. Revisit
if geometry fidelity is needed.

## 6. Verification (how the standard was checked)
- **Unit test:** for yaw ∈ {0, 90, 180, 30}° the head-camera forward equals the
  yaw-rotated base `+X` and the camera up equals odom `+Z`. Passes.
- **Pose round-trip:** `_usd_to_pose(_pose_to_usd(p)) == p` (identity, no swap). Passes.
- **Preview frame** (`preview_isaacsim_navigation.py --frames 4 --motion stationary`,
  internscenes config): head frames are ~29 KB with std ≈ 56 (real content) and the
  **bright region sits at the top / floor at the bottom** — the signature of a
  correctly-oriented forward view (the old blank frames were uniform gray, std ≈ 1.4).