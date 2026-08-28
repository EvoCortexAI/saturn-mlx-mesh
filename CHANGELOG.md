# Changelog

All notable changes to `saturn-mlx-mesh` will be documented in this file.

Versioning follows semantic-style `0.x.y` while the package remains in early development. See `Docs/VERSIONING.md` and `Docs/RELEASING.md`.

Keep active work in `Unreleased`, then promote to a dated release section when cutting a version. A changelog section is not a published tag.

## [Unreleased]

- Graph, placement, Runtime DAG, speculative, and episodic-memory research remains in-tree and is not part of the `0.2.x` Saturn-Node adapter contract.

## [0.2.0] - 2026-08-28

First published semantic release. First Apache-2.0 tagged release. Tag is created only after this section lands on `main` and founder approval records the exact SHA.

- Swift tools 6.3 / macOS 26 / iOS 26 baseline aligned with Saturn-Node.
- Stable Node adapter: `MLXInferenceRuntime`, `MeshModelInferenceRuntime`, `SimulatedMLXInferenceRuntime`.
- KF / mesh#1 model pin: `AcceptanceModelPin.primaryModelID` = `mlx-community/Qwen3-8B-4bit`.
- Opt-in hardware smoke `SaturnMLXMeshSmoke` (load, stream, cancel/recovery). Ordinary `swift test` stays weight-free.
- Product boundary: in-process MLX library only. No listener, credentials, or orchestration.
- Relicense first-party `main` materials under Apache License 2.0. Changelog `0.1.0` remains proprietary history.
- Hardware evidence for the pinned 8B-4bit path was recorded at SHA `8ce1d6f6d6f5304f526019a5b5bcbf3f2b2f783e`. This release does not reopen that gate.

See `Docs/releases/0.2.0.md`.

## [0.1.0] - 2026-06-12

- Initial private skeleton of `saturn-mlx-mesh` (Swift package for LLM inference on Apple Silicon as part of EvoIntelligenceFabric / Saturn).
- Core actor-based API: `MeshSession`, `MeshModel.generate(...)` with `speculativeGamma`.
- Placement policy system (`MeshExecutionUnit`, `PlacementDecision`, `decide(role:)`).
- Basic telemetry and error handling.
- Test simulation hook to keep CI lightweight.
- This section remains under the proprietary terms present when it was recorded. It is not a published Git tag on current Apache-2.0 `main`. Do not retag this tree as `0.1.0`.
