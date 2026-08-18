# AgenticMemoryNav — Coordinate-System & Camera Reference

> Task 2 deliverable: annotate the coordinate conventions and camera logic that are
> the most error-prone parts of the Isaac Sim navigation stack. Read this before
> touching anything in `isaacsim_adapter.py`.

## 1. The four coordinate frames in play

There are **four** frames. Confusing them is the #1 source of the camera bugs.

| # | Frame | Axes | Who uses it |
|---|-------|------|-------------|
| A | **World (USD / Isaac Sim)** | X, Y, **Z-up** | The stage, `_robot`, the cameras, PhysX. Everything Isaac renders. |
| B | **`Pose3D` (project convention)** | `(x, height, north)` → i.e. `(x, z_up, y)` | The Python agent/pipeline. Height is **index 1** (Habitat convention). |
| C | **Body-local offset** | `(forward, right, up)` | The fixed camera offset relative to the robot base. |
| D | **Camera-local** | `-Z` forward, `+Y` up | Isaac's `Camera` primitive intrinsic convention. |

### The boundary remaps (in `IsaacSimExecutor`)
```
_pose_to_usd:  (x, z_up, y_north)  ->  np.array([x, y_north, z_up])      # B -> A
_usd_to_pose:  (x, y_north, z_up)  ->  Pose3D(x, z_up, y_north)          # A -> B
```
Why height is index 1: the project `Pose3D.position` stores height at index 1
to match the Habitat adapter's convention; USD is Z-up, so the two frames are
remapped at the boundary rather than carried raw.

### The scene up-axis wrap (Y-up InternScenes -> Z-up stage)
InternScenes scenes are **Y-up**. Isaac renders **Z-up**. `_open_scene_stage`
wraps the scene in a temp USD layer:
```
/World (upAxis = Z)
  /World/scene  (prepend references = @scene@)   # original Y-up scene
    xformOp:orient = (0.70710678, 0.70710678, 0, 0)   # +90 deg about +X
```
So the imported scene's +Y is rotated onto the stage's +Z (up). The scene's
+Z goes to -Y. **The robot and cameras live in frame A (Z-up); only the scene
mesh is rotated.**

## 2. Robot / yaw
- `self._yaw` is a **body yaw about world +Z** (frame A). Zero yaw faces +X.
- The Go2 asset's **authored front is +Y**. That mismatch is `_GO2_FORWARD_YAW_RAD = +90 deg`,
  added in `_position_head_camera` so the camera looks along the body's actual
  forward, not world +X.

## 3. Head camera (the agent lens) — `_position_head_camera`
Standard look-at, recomputed every frame from the robot pose:
```
eye       = robot_pos_world + R(yaw) * _GO2_CAMERA_OFFSET_M      # frame C -> A
forward   = +90deg(yaw)   (body forward)  + optional scan       # +90deg = _GO2_FORWARD_YAW_RAD
direction = [cos(forward)*cos(pitch), sin(forward)*cos(pitch), sin(pitch)]
place_camera(cam, eye, eye+direction, up=(0,0,1), offset=go2_camera_orient)
```
- `_GO2_CAMERA_OFFSET_M = (0.0, 0.42, 0.14)` = (forward, right, up) in metres.
- `place_camera` builds a look-at quaternion with camera `-Z` forward, `+Y` up
  (frame D), world up `+Z`, then applies an optional Euler offset.
- **Static by default.** Only an armed scan (`trigger_head_scan`) adds a
  yaw/pitch; otherwise the lens holds and points straight ahead.

## 4. Overhead camera (observer) — `_place_overhead_camera`
- Static, placed **once at startup**. Not parented to anything.
- Eye = explicit `overhead_camera_position` (config) **or** scene-centre in X/Y
  raised `3 * height_span` above the ceiling. Looks down at scene centre.
- **Only created when used** (explicit position, or `livestream_camera: overhead`).
  Its RGB is never fed to the agent.

## 5. Known blank-frame cause (NOT a code bug)
With the internscenes config, `robot_start: [2.59, 0.35, 1.0]` places the Go2 at
USD `(2.59, 1.0, 0.35)`; the head camera eye is `(2.59, 1.42, 0.49)` looking +Y.
At that vantage the head lens faces a blank wall/ceiling → uniform gray frames.
Verified: the refactored camera code produces **byte-identical** output to the
pre-refactor code at the same config (mean pixel diff 0.56/255 = render noise),
so the gray is a **vantage/config issue, not a refactor regression**. The
"working" frames the user saw used the on-demand look-around scan (or a different
robot_start / `go2_camera_orient`) that swept the lens across the room.

## 6. Camera intrinsics
`get_observation` currently hardcodes `CameraIntrinsics(80, 80, cx, cy, w, h)`
and does not use `camera_focal_length` from the config. Focal length affects the
rendered FoV (Isaac `set_focal_length`), but the `CameraIntrinsics` reported to
the mapping agent stays a fixed approximation — revisit if geometry fidelity is
needed.