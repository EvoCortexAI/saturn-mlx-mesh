# Changelog

All notable changes to `saturn-mlx-mesh` will be documented in this file.

Versioning follows semantic-style `0.x.y` while the package remains in early development toward the full Saturn Mesh computation graph.

Keep active work in `Unreleased`, then promote to a dated release section when cutting a version.

## [Unreleased]

- Phase 1 (graph as first-class executable artifact): introduced `MeshComputationGraph` (L1-10) with `Device`/`SiliconExecutionUnit`/`ModelComponent` vertices, `GraphEdge` + `NodeWeight`/`EdgeWeight`, live `nodeWeights`, `active` set, `recordObservedCost` blending (from telemetry tps → lower latency/compute proxies), `l7SpeculativeExample` static builder, and internal `activate`/`addEdge` for population. `MeshSession` now owns + populates the graph in `loadModel` (and via `_registerModelForGraphTest` for CI), exposes `currentComputationGraph()`, and provides `recordObservedCostFrom` feedback. Added L7 Source/Selector/SideEffect + Phoenix-style retrieval comments in speculative/episodic paths (no logic change). New passing test `testMeshComputationGraphAndFeedback` (components, units, actives, L7 example, weight mutation after simulated fast gen). 8/8 tests green. This makes the "weighted directed computation graph" (the real IP) a concrete, queryable, updatable value — foundation for PlacementEngine rewrite, Runtime DAG executor (PlanMaster gather/merge + candidate-pipeline stages), and Epi KV residency in subsequent work.
- Introduce foundational types for the Saturn Mesh as a weighted directed computation graph (per the multi-level vision):
  - `Device`, `SiliconExecutionUnit` (Level 1–2)
  - `NodeWeight` (w(v) = (compute, memory, power, latency)) and `EdgeWeight` (w(e) = (latency, bandwidth, serialization)) (Level 9)
  - `ModelComponent` for layers, experts, embeddings, routers, etc. (Levels 3–5)
- Start EpiCache-inspired episodic memory for long conversational context (v0.2 text-level):
  - Added `EpisodicMemoryIndex` actor: utterance-window segmentation, simple trigram embedding, K-means clustering, cosine retrieval of relevant episodes.
  - Added `KVCacheManager` actor skeleton (build/retrieve/update per-episode caches; will tie to real MLX KV in v0.3).
  - Added `LayerBudgetAllocator` for future sensitivity-aware per-layer KV budgets (sharpness α ≈ 2–4 recommended).
  - Wired `EpisodicMemoryIndex` into `MeshSession` (ingest + `retrieveContext` + `generateWithMemory` that augments prompts with retrieved episode text).
  - Added basic test exercising ingest + match.
  - This enables hybrid RAG (durable facts) + episodic KV (fast conversational state) without full-history prefill. See the EpiCache-inspired design notes in the vision section below.
- Adopt the full Saturn Mesh graph vision as the guiding architecture (exact text):

  The graph for Saturn Mesh is best viewed as a weighted directed computation graph.

  Not a neural network graph.

  Not a network topology graph.

  A graph that simultaneously describes:

  * models
  * layers
  * experts
  * devices
  * execution units
  * memory
  * communication

  ⸻

  Level 1: Device Graph

  The simplest graph is:

            ┌───────────┐
            │ Saturn    │
            │ Control   │
            └─────┬─────┘
                  │
       ┌──────────┼──────────┐
       │          │          │
       ▼          ▼          ▼
   ┌──────┐   ┌──────┐   ┌──────┐
   │ Mac  │   │iPhone│   │iPad  │
   └──────┘   └──────┘   └──────┘

  Formally:

  G_D=(V_D,E_D)

  where

  V_D
  =
  \{\text{Mac},\text{iPhone},\text{iPad}\}

  and

  E_D
  =
  \text{communication links}

  Each edge has weight:

  w_e=(latency, bandwidth)

  Example:

  Mac ↔ iPhone
  latency = 2 ms
  bandwidth = 800 MB/s

  ⸻

  Level 2: Silicon Graph

  Each device expands into execution units.

  Mac:

            Mac
             │
     ┌───────┼────────┐
     │       │        │
     ▼       ▼        ▼
   CPU     GPU      ANE
   SME2

  Graph:

  G_S=(V_S,E_S)

  where

  V_S
  =
  \{
  CPU,
  GPU,
  ANE
  \}

  The important property:

  CPU
  GPU
  ANE

  share the same memory.

  Therefore:

  Cost_{transfer}
  \approx 0

  inside a device.

  This is the central Apple advantage.  

  ⸻

  Level 3: Model Graph

  Now the model itself.

  A transformer is:

  Embedding
      │
      ▼
  Layer 1
      │
      ▼
  Layer 2
      │
      ▼
  Layer 3
      │
     ...
      │
      ▼
  Layer N
      │
      ▼
  Output

  Graph:

  G_M=(V_M,E_M)

  where

  V_M
  =
  \{\text{Embedding},L_1,L_2,\ldots,L_n\}

  Edges represent tensor flow.

  ⸻

  Level 4: Placement Graph

  Now combine both.

  Each layer maps onto hardware.

  Embedding  ──► ANE
  Layer 1    ──► GPU
  Layer 2    ──► GPU
  Layer 3    ──► CPU/SME2
  Layer 4    ──► GPU

  This creates a bipartite graph:

  Model Nodes      Hardware Nodes
  Layer 1  ─────► GPU
  Layer 2  ─────► GPU
  Layer 3  ─────► CPU
  Layer 4  ─────► GPU

  Mathematically:

  P:V_M\rightarrow V_S

  Placement is simply a mapping function.

  ⸻

  Level 5: MoE Expert Graph

  For Mixture-of-Experts:

                Router
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
     Expert1    Expert2    Expert3

  Graph:

            x
            │
            ▼
         Router
            │
     ┌──────┴──────┐
     ▼             ▼
  Expert3       Expert7

  Only selected experts execute.

  Mathematically:

  A(x)
  =
  \{E_3,E_7\}

  Active subgraph:

  Router
    │
    ├──► Expert3
    │
    └──► Expert7

  The rest disappear.

  ⸻

  Level 6: Mesh Expert Graph

  Now experts live on different devices.

  Mac
   ├─ Expert 1
   ├─ Expert 2
  iPhone
   ├─ Expert 3
  iPad
   ├─ Expert 4

  Graph:

              Router
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
       E1       E3       E4
       │        │        │
      Mac    iPhone    iPad

  Now routing becomes graph traversal.

  ⸻

  Level 7: Speculative Graph

  Drafter + verifier.

  Prompt
     │
     ▼
  Drafter
     │
     ▼
  Candidate Tokens
     │
     ▼
  Verifier
     │
     ▼
  Accepted Tokens

  Graph:

             Prompt
                │
                ▼
           Drafter
                │
        Candidate Path
                │
                ▼
            Verifier
                │
                ▼
            Output

  Potentially:

  iPhone ANE
        │
        ▼
     Drafter
        │
        ▼
  Mac GPU
        │
        ▼
   Verifier

  This is probably the first truly useful cross-device graph.

  ⸻

  Level 8: Runtime DAG

  The actual Saturn graph becomes:

            Prompt
                │
                ▼
            Router
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
   Drafter   Embedder   Vision
      │         │         │
      └────┬────┴────┬────┘
           ▼         ▼
        Primary LLM
              │
              ▼
          Reranker
              │
              ▼
           Output

  This is no longer a transformer.

  It is a distributed AI system.

  ⸻

  Level 9: Weighted Graph

  Each node gets:

  compute cost
  memory cost
  power cost
  latency

  Node weight:

  w(v)
  =
  (c,m,p,t)

  Each edge gets:

  network latency
  serialization cost
  bandwidth

  Edge weight:

  w(e)
  =
  (l,b,s)

  Now the scheduler solves:

  \min
  \sum w(v)
  +
  \sum w(e)

  subject to quality constraints.

  ⸻

  Level 10: The Full Saturn Graph

  Conceptually:

                   User
                     │
                     ▼
                MeshSession
                     │
              PlacementEngine
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
   Mac GPU      iPhone ANE     iPad CPU
      │              │              │
      ▼              ▼              ▼
   Experts      Drafters      Embedders
      │              │              │
      └───────┬──────┴──────┬───────┘
              ▼             ▼
                Primary LLM
                      │
                      ▼
                   Output

  In graph-theory terms, Saturn eventually becomes a dynamic weighted heterogeneous execution DAG where:

  * vertices = models, layers, experts, execution units
  * edges = tensor movement and communication
  * weights = latency, power, memory, bandwidth
  * scheduler = optimizer that continuously rewrites the graph

  That graph—not the transformer itself—is the real intellectual property opportunity behind EvoIntelligenceFabric and the Saturn mesh vision.
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