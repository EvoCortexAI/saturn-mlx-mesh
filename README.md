# saturn-mlx-mesh

**Mesh LLM extension for MLX on Apple Silicon**

Part of [EvoIntelligenceFabric](https://evocortex.ai) — the execution plane (Saturn-Node).

The package is built toward the vision of **Saturn Mesh as a weighted directed computation graph** (not a neural network graph or network topology). This graph simultaneously describes models, layers, experts, devices, execution units (CPU/GPU/ANE under UMA), memory, and communication, with node weights w(v)=(compute, memory, power, latency) and edge weights w(e)=(latency, bandwidth, serialization). The scheduler optimizes min ∑w(v) + ∑w(e) subject to quality constraints. See the full vision in CHANGELOG.md.

## Public API (exact contract)

```swift
let mesh = MeshSession(controlPlane: .local, policy: .appleSiliconBalanced)
let model = try await mesh.loadModel(id: "mlx-community/Qwen3-8B-4bit", role: .primary)
let stream = try await model.generate(
    prompt: "Explain Saturn mesh inference.",
    maxTokens: 512,
    temperature: 0.7,
    speculativeGamma: 4   // optional
)
for try await token in stream {
    print(token.text, terminator: "")
}
```

## Key Components

- `MeshSession` — actor factory, control plane selection, policy
- `MeshModel` — actor-isolated, generation (standard + speculativeGamma surface with real drafter usage), KV cache reuse via `TokenIterator` + `KVCacheBox`, telemetry hooks
- `PlacementPolicy` + `AppleSiliconBalanced` — `MeshExecutionUnit`, `PlacementDecision`, `decide(role:)` with explicit `NodeWeight`/`EdgeWeight` cost model (w(v) + w(e))
- `MeshTelemetry` — actor for load + generation records (truthful speculative fields)
- Graph foundation types (`Device`, `SiliconExecutionUnit`, `NodeWeight`, `EdgeWeight`, `ModelComponent`) modeling the multi-level computation graph (Device/Silicon/Model/Placement/MoE/Mesh/Speculative/Runtime DAG/Weighted/Full)

## Status (progressing toward full weighted computation graph)

- ✅ Exact public API surface (MeshSession + MeshModel are actors)
- ✅ Real model loading wired via `LLMModelFactory` + Hub downloader/tokenizer loader macros (with required `HuggingFace`/`Tokenizers` products)
- ✅ KV cache reuse implemented for main generation path (explicit `TokenIterator` + persistent cache via `KVCacheBox`)
- ✅ Drafter loading: speculative path actually executes the attached drafter `ModelContainer` for proposals
- ✅ Graph foundation types + explicit weighted cost model in placement (`NodeWeight` w(v), `EdgeWeight` w(e), cost-driven `placementCost`)
- ✅ Placement policy wired + recorded in telemetry; decisions now report formal graph costs
- ✅ Actor isolation throughout
- ✅ Telemetry (truthful: speculative fields only when drafter attached *and* used)
- ✅ Stream completion fixed (`continuation.finish()` on success)
- ✅ Basic unit tests + coverage for generate behavior (.notLoaded, stream finishes, no-spec telemetry)
- ✅ Opt-in hardware smoke executable (`SaturnMLXMeshSmoke`) for real inference (excluded from `swift test`)
- ✅ v0.2 EpisodicMemoryIndex (text-level): utterance segmentation, embedding, K-means episodes, retrieval for long-context augmentation. Integrated into MeshSession.
- Simulation hook (`_enableTestSuccessSimulation()`) preserved so unit tests/CI require no model weights or GPU
- KVCacheManager and LayerBudgetAllocator skeletons added (real MLX KV compression + layer budgets in v0.3+)
- See [CHANGELOG.md](CHANGELOG.md) for the full adopted graph vision (10 levels: Device Graph → Full Saturn Graph) and detailed milestone history.

See [CHANGELOG.md](CHANGELOG.md) for the full adopted Saturn Mesh graph vision (the 10-level weighted directed computation graph) and detailed history. See [Docs/mesh-llm-mlx-extension.md](Docs/mesh-llm-mlx-extension.md) for additional rollout notes and math references (placement cost model, speculative speedup formula).

## Build & Test

```bash
swift build
swift test
```

Requires Apple Silicon (macOS 15+ / iOS 18+ target) for real MLX execution. The unit tests are designed to pass without downloading large models.

## Dependencies

- `mlx-swift-lm` (MLXLLM + MLXLMCommon + MLXHuggingFace)
- `swift-huggingface` (HuggingFace module for downloader)
- `swift-transformers` (Tokenizers for tokenizer loader)

## Relationship to the Rest of Saturn

- **Control plane**: `saturn-control` (orchestration, node registry, scheduling, OpenAI-compatible gateway). Never runs heavy inference.
- **Execution plane**: This package + surrounding Saturn-Node runtime on Apple Silicon devices (primary/secondary/drafter roles, UMA placement, power telemetry).
- **Client plane**: Saturn One (Apple app).

Nodes register capabilities with labels such as `["mlx", "primary"]`. This library powers the actual inference work on those nodes.

## Next (aligned with the graph vision + EpiCache memory)

- v0.3: Real MLX KV cache hooks + block-wise prefill + per-episode `EpisodeKVCache` objects + KVCacheManager integration.
- v0.4: Sensitivity-aware layer budget allocation (LayerBudgetAllocator) when building compressed episode caches.
- Full verifier/drafter speculative acceptance (already partially wired; complete propose + verify path with cache rollback).
- Cross-device support (Device Graph, Mesh Expert Graph) and remote `ControlPlane`.
- Runtime DAG representation (Router, Embedder, Vision, etc.) and basic `MeshKVCache` / scheduler that optimizes min ∑w(v) + ∑w(e).
- Real-hardware validation + hybrid RAG + episodic KV memory in production flows.
- Expand to full MeshKVCache abstraction: full session + episodic caches + layer policies + remote residency + refresh protocol.

## License / Copyright

Copyright © 2026 EvoCortexAI S.L. All rights reserved.

---

This repository is intentionally separate from `saturn-control`. It is the MLX-native mesh execution component for private on-device inference in the EvoIntelligenceFabric.