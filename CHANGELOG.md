# Changelog

All notable changes to `saturn-mlx-mesh` will be documented in this file.

Versioning follows semantic-style `0.x.y` while the package remains in early development toward the full Saturn Mesh computation graph.

Keep active work in `Unreleased`, then promote to a dated release section when cutting a version.

## [Unreleased]

- Introduce foundational types for the Saturn Mesh as a weighted directed computation graph (per the multi-level vision):
  - `Device`, `SiliconExecutionUnit` (Level 1–2)
  - `NodeWeight` (w(v) = (compute, memory, power, latency)) and `EdgeWeight` (w(e) = (latency, bandwidth, serialization)) (Level 9)
  - `ModelComponent` for layers, experts, embeddings, routers, etc. (Levels 3–5)
- Enhance `AppleSiliconBalanced` placement policy with explicit weighted cost model:
  - `nodeWeight(for:)` returning `NodeWeight`
  - `placementCost(for:unit:maxKVSizeHint:)` = base (from w(v)) + rolePenalty + kvPenalty
  - `decide(role:)` now reports the formal graph cost in `PlacementDecision.notes` (while preserving v0.1 role-based unit selection for compatibility)
  - Intra-device edges explicitly noted as ~0 cost due to UMA (Level 2 advantage)
- Complete "drafter loading" step: when `drafterId` is supplied to `loadModel` and `speculativeGamma` is requested, the speculative branch now actually executes inference on the attached drafter `ModelContainer` (using `TokenIterator` for consistency with the main path). Updated `_runSpeculative` to prepare input inside the drafter's `perform` closure (avoids non-Sendable capture issues). Full verifier + acceptance logic remains the subsequent milestone.
- Implement KV cache reuse for the main (non-speculative) generation path:
  - Added internal `KVCacheBox` (`@unchecked Sendable` holder) to safely bridge `[any KVCache]` across the `Sendable` `ModelContainer.perform` closure.
  - Cache created on first use via `context.model.newCache` (respecting `placement.maxKVSizeHint` when present) and reused on subsequent `generate()` calls on the same `MeshModel`.
  - Replaced high-level `MLXLMCommon.generate` + chunk loop with explicit `TokenIterator` + decode + yield in the standard path.
  - Simulation hook and speculative fallback left unchanged per the defined sequence.
- Narrow real-loading first pass (the "right next checkpoint"):
  - Wired real `LLMModelFactory.shared.loadContainer(from: #hubDownloader(), using: #huggingFaceTokenizerLoader(), configuration:)` using the current `mlx-swift-lm` API.
  - Added required products in `Package.swift` (`HuggingFace` from `swift-huggingface`, `Tokenizers` from `swift-transformers`) alongside existing MLXLLM/MLXLMCommon/MLXHuggingFace.
  - Added opt-in hardware smoke executable target `SaturnMLXMeshSmoke` (never depended on by tests/CI).
  - Preserved `_enableTestSuccessSimulation()` hook so normal `swift test` requires no model weights or GPU.
- Added focused unit tests for generate behavior (skeleton-hardening):
  - `testGenerateThrowsNotLoadedWhenNoContainerAttached`
  - `testStreamFinishesAfterSuccessfulGeneration` (verifies `continuation.finish()`; uses simulation hook)
  - `testSpeculativeTelemetryAbsentWhenNoDrafterAttached` (verifies truthful telemetry)
- Aligned documentation with reality:
  - Updated `README.md` and `Docs/mesh-llm-mlx-extension.md` to accurately describe stubbed loader (at the time), high-level generation path, lack of full KV reuse and speculative acceleration, simulation hook for tests, etc.
  - Explicitly reference the graph vision and the recommended milestone order (real loading → KV reuse → drafter loading → verifier/drafter acceptance).
- Fixed stream completion: `AsyncThrowingStream` now calls `continuation.finish()` on the success path after telemetry (previously callers could hang).
- Made telemetry truthful:
  - `speculativeGamma` / `acceptedTokens` are only populated in `GenerationInfo` when a drafter was attached *and* the speculative branch was actually taken.
  - Added notes that `generatedTokens` currently counts generator chunks (not raw tokens).
- Initial v0.1 skeleton (pre-graph work):
  - `MeshSession` and `MeshModel` as true actors.
  - `PlacementPolicy` / `AppleSiliconBalanced` with `MeshExecutionUnit`, `PlacementDecision`, `ModelRole`.
  - `MeshTelemetry` actor with load/generation records and `snapshot()`.
  - Real MLX loading via `mlx-swift-lm` (with test simulation path).
  - Basic streaming generation + early speculative surface (high-level fallback).
  - `Package.swift` with correct MLX dependencies; opt-in smoke; clean structure.

## [0.1.0] - 2026-06-12

- Initial public skeleton of `saturn-mlx-mesh` (Swift package for efficient, heterogeneous LLM inference on Apple Silicon as part of EvoIntelligenceFabric / Saturn).
- Core actor-based API: `MeshSession`, `MeshModel.generate(...)` with `speculativeGamma`.
- Placement policy system (`MeshExecutionUnit`, `PlacementDecision`, `decide(role:)`).
- Basic telemetry and error handling.
- Test simulation hook to keep CI lightweight.
- Real-loading, KV reuse, drafter usage, and graph cost model work landed in subsequent unreleased increments on the path to the full weighted computation graph.