# Limitations

- The shipped mapper and perception agents are deterministic mocks. Their successful
  run validates system integration, not reconstruction or semantic accuracy.
- LingBot-Map has been smoke-tested on the local RTX A6000 using Python 3.10, CUDA
  PyTorch 2.8.0, the vendored `lingbot-map.pt` checkpoint, and two local RGB frames.
  It produced depth, pose, and point-head `world_points`; this verifies runtime
  execution only, not reconstruction quality. The point-head checkpoint load reported
  62 missing keys, so its geometry must not be treated as calibrated ground truth.
- Point-cloud comparisons require paired metric ground truth and a verified shared
  camera-to-world convention. The default evaluator intentionally performs no alignment;
  centroid alignment is diagnostic and can conceal global translation error.
- A 24-frame Stage1/VGGT depth-plus-pose diagnostic showed that Sim(3) trajectory
  alignment can reduce absolute trajectory error while leaving point-cloud geometry
  inaccurate. Similarity alignment is therefore a calibration diagnostic, not a
  geometry-accuracy correction; submaps must retain their raw provenance and be
  cross-checked against depth/LiDAR or paired GT before becoming high-confidence facts.
- A 200-frame Isaac GT depth/c2w sequence validated the local-submap overlap gate
  itself: its stable 200-frame submap reproduced GT geometry exactly. The selected
  runtime policy is RGB-only: stable Stage1/VGGT depth-plus-pose windows are written to
  spatial memory based on internal overlap consistency, with provenance and confidence.
  Their metric depth scale can still drift, so downstream reasoning must preserve that
  uncertainty. LiDAR and GT are optional offline evaluation sources, not required
  runtime verifiers; the recorded 2D LiDAR is not yet calibrated for automatic fusion.
- `scripts/record_isaacsim_sequence.py` has been runtime-verified for a three-frame
  procedural PointNav-style scene: RGB, metric depth, $4\times4$ c2w poses, and 3,600
  RTX 2D LiDAR scan samples per frame were emitted, and the depth/pose sequence produced
  a finite 6,520-point GT cloud. The recorder's quadruped is a kinematic visual proxy,
  not a physically simulated dog-shaped robot or Unitree Go2; the `Example_Rotary_2D`
  LiDAR is also a generic simulator profile rather than a hardware-calibrated sensor.
- The Habitat adapter is a boundary stub. Habitat can provide RGB-D, NavMesh, path, and
  collision state, but this project does not claim Go2 gait or contact fidelity.
- The Isaac Sim adapter (`IsaacSimExecutor`/`IsaacSimObjectNavExecutor`) moves the robot
  kinematically (direct pose interpolation), like the Habitat and Unitree-sim adapters.
  It never reports a real collision (`is_collision()` is always `False`), so PointNav SPL
  results reflect direct-line travel distance, not obstacle-aware navigation.
- PointNav episodes are procedurally generated over a fixed obstacle layout, not the
  official Habitat/HM3D/MP3D PointNav datasets; results are not paper-comparable.
- ObjectNav (`IsaacSimObjectNavExecutor`) requires the InteriorAgent dataset, which was
  not available in this environment; the executor and script are implemented but have
  not been runtime-verified end to end. Its `shortest_distance_to_goal()` is a Euclidean
  lower bound, not a validated geodesic/occupancy-aware distance.
- `UnitreeSimExecutor` models planar base movement only. It does not model joints,
  balance, terrain contact, latency, or real actuator limits.
- Local hashed text vectors are deterministic retrieval fallbacks, not semantic language
  embeddings. A vector database can replace them through the storage boundary.
- Rule-based parsing covers the MVP instruction family. It is not a general language
  understanding system.
- Scene relations are geometric heuristics and can be mutually redundant. Production
  use needs relation conflict resolution and calibrated uncertainty.
- Current metrics include runnable core formulas and artifact consistency checks. Full
  benchmark evaluation requires external ground truth, calibration, and dataset assets.
- Real robot execution remains disabled. Safety behavior has not been certified for any
  physical platform.