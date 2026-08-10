# Limitations

- The shipped mapper and perception agents are deterministic mocks. Their successful
  run validates system integration, not reconstruction or semantic accuracy.
- LingBot-Map activation is intentionally gated until camera-to-world pose convention,
  checkpoint licensing, CPU behavior, and state replay are tested on known trajectories.
- The Habitat adapter is a boundary stub. Habitat can provide RGB-D, NavMesh, path, and
  collision state, but this project does not claim Go2 gait or contact fidelity.
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