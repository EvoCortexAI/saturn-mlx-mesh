# Saturn MLX Mesh — LLM Extension for MLX on Apple Silicon

**Package**: `saturn-mlx-mesh` (SwiftPM library `SaturnMLXMesh`)  
**Status**: Progressing toward full weighted directed computation graph + episodic long-term memory. Real loading, KV reuse, drafter paths, graph types, and v0.2 text-level episodic memory implemented.  
**Part of**: EvoIntelligenceFabric — Execution plane (Saturn-Node)

## Vision: Saturn Mesh as a Weighted Directed Computation Graph

The core insight (from the project vision) is that the Saturn Mesh is best viewed as a **weighted directed computation graph** — not a neural network graph and not a network topology graph.

This graph simultaneously describes:

* models
* layers
* experts
* devices
* execution units
* memory
* communication

### Level 1: Device Graph
The simplest graph connects control plane to devices (Mac, iPhone, iPad, etc.).

Formally: G_D = (V_D, E_D) where edges carry w_e = (latency, bandwidth).

### Level 2: Silicon Graph
Each device expands into execution units (CPU, GPU, ANE/SME2) that share unified memory (UMA). Intra-device transfer cost ≈ 0 — the central Apple advantage.

### Level 3: Model Graph
Transformers and other models as graphs of components (Embedding → Layer 1 … Layer N → Output). Edges represent tensor flow.

### Level 4: Placement Graph
Bipartite mapping P: V_M → V_S of model components onto hardware units.

### Level 5–6: MoE / Mesh Expert Graph
Experts (in MoE) or model shards placed across devices. Routing becomes graph traversal. Cross-device mesh (Mac ↔ iPhone ↔ iPad) with real communication costs.

### Level 7: Speculative Graph
Drafter + verifier paths (prompt → drafter proposals → verifier acceptance → output). Can span devices (e.g., iPhone ANE drafter + Mac GPU verifier).

### Level 8: Runtime DAG
The actual execution is a rich DAG: Prompt → Router → (Drafter | Embedder | Vision) → Primary LLM → Reranker → Output. Not just a single transformer — a distributed AI system.

### Level 9: Weighted Graph
Every node has w(v) = (compute, memory, power, latency).
Every edge has w(e) = (latency, bandwidth, serialization).
The scheduler solves: min ∑w(v) + ∑w(e) subject to quality constraints.

### Level 10: The Full Saturn Graph
MeshSession + PlacementEngine operate over a dynamic weighted heterogeneous execution DAG. Vertices = models/layers/experts/units; edges = tensor movement/comms; weights as above; the scheduler continuously rewrites the graph.

**That graph — not the transformer itself — is the real intellectual property opportunity.**

See also the full vision text in CHANGELOG.md (adopted as guiding architecture) and the EpiCache-inspired episodic memory work below.

This package delivers the concrete MLX implementation on Apple Silicon (UMA placement, real loading, KV reuse, speculative paths, etc.) while the broader Saturn Control plane orchestrates the mesh.

## Current Scope (graph-aware + episodic memory)

- Single-process Apple Silicon focus for now (cross-device / multi-node via the graph model and future remote ControlPlane).
- `MeshSession` + `MeshModel` are true actors. Public API matches the design (loadModel with role/drafterId, generate with speculativeGamma, telemetry snapshot, etc.).
- Model loading is fully wired via `LLMModelFactory` + `#hubDownloader` / `#huggingFaceTokenizerLoader` macros (plus required HuggingFace + Tokenizers products). Real execution requires Apple Silicon + model weights.
- Main generation path uses explicit `TokenIterator` + reusable KV cache (via `KVCacheBox`) for efficiency across turns.
- Speculative path supports drafter loading; full propose + verifier acceptance/rejection with cache rollback is implemented in the current step (see code and prior "next steps").
- Placement uses the graph foundation: `Device`, `SiliconExecutionUnit`, `NodeWeight` (w(v) = (c,m,p,t)), `EdgeWeight` (w(e)), `ModelComponent`. `AppleSiliconBalanced` policy now computes and reports formal weighted costs (Level 9).
- **EpiCache-inspired episodic memory (v0.2 text-level)**: `EpisodicMemoryIndex` actor (utterance-window segmentation, embedding, K-means clustering, retrieval), `KVCacheManager` and `LayerBudgetAllocator` skeletons. Integrated into `MeshSession` (ingest + `retrieveContext` + `generateWithMemory`). Text/RAG layer today; real per-episode compressed KV + layer budgets in v0.3+.
- Telemetry is truthful (speculative fields only when drafter actually used). Stream completion is correct.
- Opt-in hardware smoke (`SaturnMLXMeshSmoke` executable) for real inference; unit tests use simulation hook so CI never needs weights/GPU.
- Comments and docs reference the full multi-level computation graph vision, placement cost model, and speculative speedup formula.

**Current limitations (intentional for incremental delivery):**
- Full cross-device mesh and remote execution are modeled in the graph vision but not yet runtime-wired.
- EpiCache KV compression (block prefill, real episode caches, sensitivity-aware budgets) is stubbed for v0.3.
- Complete Runtime DAG support (routers, vision, rerankers as first-class components) is evolving.
- Real multi-node / distributed scheduling is future work.

## Exact Public API (must remain stable)

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

Additional surface (PlacementPolicy, TelemetrySnapshot, etc.) is public for diagnostics and future control-plane integration.

## Files in this package

```
saturn-mlx-mesh/
├── Package.swift
├── Sources/SaturnMLXMesh/
│   ├── MeshSession.swift          // actor factory + policy + episodic memory integration
│   ├── MeshModel.swift            // actor-isolated model (real loading, TokenIterator KV reuse, speculative acceptance)
│   ├── PlacementPolicy.swift      // graph types (Device, SiliconExecutionUnit, NodeWeight/EdgeWeight, ModelComponent) + weighted cost policy
│   ├── MeshTelemetry.swift        // truthful load/generation records
│   ├── EpisodicMemoryIndex.swift  // v0.2 text-level episodic memory (segment/embed/K-means/retrieve)
│   ├── KVCacheManager.swift       // skeleton for per-episode KV caches (v0.3+)
│   ├── LayerBudgetAllocator.swift // sensitivity-aware layer KV budgets (v0.4+)
│   ├── MeshExecutionUnit.swift    // small re-export
│   └── (minimal internal helpers)
├── Sources/SaturnMLXMeshSmoke/
│   └── main.swift                 // opt-in real-inference smoke (excluded from swift test)
├── Tests/SaturnMLXMeshTests/
│   └── SaturnMLXMeshTests.swift   // policy, telemetry, generate behavior, episodic memory
└── Docs/
    └── mesh-llm-mlx-extension.md  // this file
```

## Dependencies

- `mlx-swift-lm` (provides `MLXLLM`, `MLXLMCommon`, and `MLXHuggingFace`)
- `swift-huggingface` (HuggingFace module + macro support for downloader)
- `swift-transformers` / `Tokenizers` (tokenizer loader + macro support)
- Targets macOS 15+ / iOS 18+

## Validation performed

```bash
swift build
swift test
```

(Includes the full test suite plus targeted coverage for episodic memory index, graph cost modeling, speculative acceptance, stream completion, etc.)

Real inference, long-context memory, and hardware behavior are exercised via the opt-in `SaturnMLXMeshSmoke` executable (excluded from normal `swift test` / CI so no weights are required for automated runs).

## Recommended next steps (aligned with graph vision + EpiCache)

1. v0.3: Real MLX KV cache hooks, block-wise prefill compression, per-episode `EpisodeKVCache` objects, and full `KVCacheManager` integration.
2. v0.4: Wire `LayerBudgetAllocator` for sensitivity-aware per-layer KV budgets when building compressed episode caches.
3. Complete / harden full verifier/drafter speculative acceptance (propose + verify/accept/reject + rollback) and integrate with episodic memory.
4. Cross-device Device Graph / Mesh Expert Graph support and remote `ControlPlane` execution.
5. Runtime DAG modeling (Router, Embedder, Vision, Reranker as first-class `ModelComponent`s) + basic `MeshKVCache` / graph scheduler that optimizes min ∑w(v) + ∑w(e) using live telemetry.
6. Real-hardware validation, hybrid RAG + episodic KV flows, and expansion of the smoke/benchmark targets.

The package is intentionally separate from saturn-control (execution plane only). The graph—not any single transformer—is the central abstraction.

## Rollback / safety

- Execution-plane code: failures are isolated to the node running inference.
- Saturn-Control treats execution nodes as opaque (via typed contracts and the `/v1` gateway).
- Rollback is node-local (different binary / config on the Saturn-Node device).

## References

- See the root README.md of this repository for architecture context.
- Original design prompt + council review (Atlas, Forge, Glock, Cipher) covering UMA placement, speculative decoding math, KV cache reuse, and strict v0.1 minimal scope.
- Related control-plane docs live in the separate `saturn-control` repository (STRATEGY.md, ARCHITECTURE.md, CONTROL-NODE-API.md, etc.).

---

*Generated as the complete initial codebase for the reviewed prompt. Prioritizes correctness, actor isolation, cache reuse, and a verifiable speculative fallback.*
