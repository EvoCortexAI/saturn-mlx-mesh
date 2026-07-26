# saturn-mlx-mesh

**MLX-native inference library for Apple Silicon**

Part of EvoIntelligenceFabric and the Saturn-Node execution plane.

This package is built toward Saturn Mesh as a weighted directed computation graph. The long-term graph may describe models, layers, experts, devices, execution units, memory, communication, and placement cost. That research direction remains valid, but it is not the immediate MVP gate.

## MVP Priority - One Reliable Local Inference Path

The current priority is to prove one repeatable, real-hardware inference path on Saturn-Node-01 before expanding distributed mesh behavior.

```text
Saturn One -> Saturn-Control -> Saturn-Node runtime -> saturn-mlx-mesh / MLX model
```

This repository owns the **inference library**. It does not own:

- the Saturn One user interface;
- the Saturn-Control client-facing API;
- node authentication or public network exposure;
- service installation, launchd lifecycle, health endpoints, or operational rollback.

Those runtime and operational responsibilities should live in a thin, private `saturn-node` service repository. Saturn-Control tracks creation of that repository until it exists.

### MVP acceptance sequence

1. Select and pin one model/runtime combination for SN01 after checking available memory, cached weights, and real hardware behavior.
2. Complete a real model-load and streamed-generation smoke run from a documented clean state.
3. Verify token streaming, explicit cancellation, timeout behavior, and recovery after client disconnect.
4. Verify managed restart followed by successful inference without ad hoc repair.
5. Record model ID, runtime revision, node identity, timing metadata, and outcome without logging prompt or generated-response content by default.
6. Repeat the fixed acceptance prompt set before changing placement, speculative, episodic-memory, or graph behavior.

`mlx-community/Qwen3-8B-4bit` is an existing smoke-test candidate, not a release commitment. The selected model must be confirmed from actual SN01 hardware evidence.

### Deferred until the MVP gate is green

- Multi-device execution and remote control-plane placement
- Automatic multi-node routing
- New Runtime DAG expansion
- Additional scheduler complexity
- New speculative-decoding research beyond what the selected path requires
- Broader episodic-memory and hybrid RAG product integration

## Public API

```swift
let mesh = MeshSession(controlPlane: .local, policy: .appleSiliconBalanced)
let model = try await mesh.loadModel(
    id: "mlx-community/Qwen3-8B-4bit",
    role: .primary
)
let stream = try await model.generate(
    prompt: "Explain Saturn mesh inference.",
    maxTokens: 512,
    temperature: 0.7,
    speculativeGamma: 4
)
for try await token in stream {
    print(token.text, terminator: "")
}
```

The example demonstrates the current library surface. Production model selection, limits, and routing belong to the managed Saturn-Node runtime and Saturn-Control policy.

## Key Components

- `MeshSession` - actor factory, control-plane selection, and placement policy
- `MeshModel` - actor-isolated standard and speculative generation, KV-cache reuse, and telemetry hooks
- `PlacementPolicy` and `AppleSiliconBalanced` - execution-unit decisions with explicit node and edge cost concepts
- `MeshTelemetry` - load and generation records
- Graph foundation types for devices, silicon execution units, model components, weights, and episode residency
- `KVCacheManager`, `LayerBudgetAllocator`, and `EpisodeKVCache` for cached-prefix and episodic-memory work

## Current Technical State

- Exact `MeshSession` and `MeshModel` actor-based API surface
- Real model loading through MLX Swift LM and Hugging Face integrations
- KV-cache reuse in the main generation path
- Drafter-model loading and speculative-generation surfaces
- Weighted graph and placement foundations
- Actor isolation and stream completion handling
- Telemetry that distinguishes actual speculative use
- Opt-in Apple Silicon hardware smoke executable
- Text-level episodic-memory indexing and retrieval
- Per-episode KV-cache, allocation, and graph-residency foundations
- Simulation hook so normal unit tests do not download model weights or require a GPU

The technical foundations are broader than the MVP. A feature being present in this package does not make it part of the first Saturn product or local-inference release gate.

## Build and Test

```sh
swift build
swift test
```

Real MLX execution requires compatible Apple Silicon. Unit tests are designed to run without downloading large model weights.

Run the opt-in hardware smoke manually on the target node:

```sh
swift run SaturnMLXMeshSmoke
```

The smoke result is not sufficient by itself. The MVP also requires cancellation, restart, private-service, and end-to-end Saturn-Control verification.

## Dependencies

- `mlx-swift-lm` (`MLXLLM`, `MLXLMCommon`, and `MLXHuggingFace`)
- `swift-huggingface` (`HuggingFace` downloader integration)
- `swift-transformers` (`Tokenizers`)

Dependency revisions used by the managed node must be pinned and recorded in its runtime manifest. Do not treat a floating package range as a production deployment record.

## Relationship to the Rest of Saturn

- **Client plane:** Saturn One, the Apple-native user interface and control surface
- **Control plane:** Saturn-Control, responsible for orchestration, policy coordination, request correlation, and the client-facing API; never heavy inference
- **Execution service:** Saturn-Node, a private managed process that owns service lifecycle, authentication, health, and API adaptation
- **Inference library:** this package, used by Saturn-Node for MLX-native execution
- **Governance:** EvoEthics, introduced after the local-inference contract is proven and approved

Nodes may register capabilities such as `mlx` and `primary`, but automatic multi-node scheduling is outside the MVP.

## Immediate Next Work

1. Create or approve the thin `saturn-node` runtime repository and its API/lifecycle boundary.
2. Pin one SN01 runtime and candidate model in a machine-readable manifest.
3. Run the real-hardware smoke and save reproducible metadata.
4. Add fixed tests for stream completion, cancellation, disconnect, timeout, and restart behavior at the runtime boundary.
5. Validate the complete Saturn-Control gateway path before resuming distributed graph work.

The system-wide delivery order and acceptance thresholds are maintained in `saturn-control/docs/SATURN-MVP.md`.

## Long-Term Research Direction

After the single-node gate is reliable, the package may resume work on:

- hardened verifier/drafter acceptance and rollback;
- cross-device Device Graph and Mesh Expert Graph support;
- Runtime DAG modeling for routers, embedders, vision, and rerankers;
- telemetry-driven graph scheduling and residency-aware placement;
- expanded episodic-memory, hybrid RAG, smoke, and benchmark targets.

Long-term work must remain separable from the stable local-inference contract used by Saturn One and Saturn-Control.

## License and Copyright

Copyright © 2026 EvoCortexAI S.L. All rights reserved.

This repository is intentionally separate from Saturn-Control. It is the MLX-native inference component for private execution in EvoIntelligenceFabric, not the control plane or a standalone user-facing product.
