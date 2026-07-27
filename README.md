# saturn-mlx-mesh

**MLX-native inference library for Apple Silicon**

Part of EvoIntelligenceFabric and loaded inside Saturn-Node.

This package is not Saturn-Node itself. It is the in-process MLX execution library used by the private Saturn-Node inference service.

## Canonical runtime path

```text
Saturn One or Saturn Container
    -> Saturn-Control
    -> Saturn-Control Container Runner
    -> Apple Container agent workload
    -> workload-authenticated Saturn-Node
    -> saturn-mlx-mesh
    -> MLX model
```

Frontends call Saturn-Control only. Agent containers call their assigned Saturn-Node with a short-lived compute credential issued by Saturn-Control.

This repository owns:

- MLX model loading and inference primitives;
- token streaming;
- cancellation hooks;
- KV-cache behavior;
- placement and graph research;
- inference telemetry primitives;
- deterministic simulation and hardware smoke support.

It does not own:

- Apple Container execution or lifecycle;
- Saturn-Control orchestration;
- agent logic or tools;
- workload credential issuance or verification;
- a network listener or public API;
- service installation, launchd lifecycle, health endpoints, or rollback;
- user authentication;
- frontend behavior.

Those service responsibilities belong in a separate private `saturn-node` repository. Container execution belongs to Saturn-Control's Container Runner.

## MVP priority - one reliable inference path

The immediate priority is one repeatable, real-hardware inference path on Saturn-Node-01:

```text
agent container -> Saturn-Node -> saturn-mlx-mesh -> pinned MLX model
```

### Acceptance sequence

1. Pin one MLX runtime revision and model manifest after checking target hardware.
2. Complete a clean-state model-load and streamed-generation smoke.
3. Verify token streaming, explicit cancellation, timeout, and disconnect recovery.
4. Verify managed Saturn-Node restart followed by successful inference.
5. Verify model, context, output, concurrency, and budget enforcement at the service boundary.
6. Record model, runtime, node, timing, token-count, and outcome metadata without prompt or response content.
7. Repeat a fixed acceptance set before expanding graph, placement, speculative, or episodic-memory behavior.

`mlx-community/Qwen3-8B-4bit` is an existing smoke-test candidate, not a release commitment.

### Deferred

- multi-device execution;
- automatic multi-node routing;
- new Runtime DAG expansion;
- additional scheduler complexity;
- speculative-decoding research beyond the selected path;
- broad episodic-memory and hybrid RAG integration;
- service authentication or orchestration code in this package;
- Apple Container integration.

## Public library API

```swift
let mesh = MeshSession(
    controlPlane: .local,
    policy: .appleSiliconBalanced
)

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

This is an in-process library surface. Production model selection, workload authentication, quotas, request validation, and network streaming belong to Saturn-Node. Policy and compute assignment belong to Saturn-Control.

## Key components

- `MeshSession` - actor factory, local control-plane selection, and placement policy
- `MeshModel` - actor-isolated generation, KV-cache reuse, and telemetry hooks
- `PlacementPolicy` and `AppleSiliconBalanced` - execution-unit decisions with explicit cost concepts
- `MeshTelemetry` - model-load and generation records
- graph foundation types for devices, execution units, model components, weights, and residency
- `KVCacheManager`, `LayerBudgetAllocator`, and `EpisodeKVCache`

The library's `controlPlane: .local` value is internal inference-library configuration. It does not mean Saturn-Control or Apple Container orchestration lives in this package.

## Current technical state

- actor-based `MeshSession` and `MeshModel`
- real model loading through MLX Swift LM and Hugging Face integrations
- KV-cache reuse in the main generation path
- drafter loading and speculative-generation surfaces
- weighted graph and placement foundations
- stream completion and cancellation-oriented structure
- telemetry that distinguishes actual speculative use
- opt-in Apple Silicon hardware smoke executable
- episodic-memory indexing and retrieval foundations
- simulation hook so unit tests avoid model downloads and GPU requirements

The technical foundation is broader than the MVP. A feature in this package is not automatically a supported Saturn product capability.

## Build and test

```sh
swift build
swift test
```

Real MLX execution requires compatible Apple Silicon. Run the opt-in hardware smoke manually on the target:

```sh
swift run SaturnMLXMeshSmoke
```

The smoke alone is insufficient. Saturn-Node must also prove workload authentication, limits, streaming, cancellation, restart, and end-to-end agent-container behavior.

## Dependencies

- `mlx-swift-lm`
- `swift-huggingface`
- `swift-transformers`

The managed Saturn-Node runtime must pin and record deployed dependency revisions. A floating SwiftPM range is not a production manifest.

## Saturn-Node integration

See [`Docs/SATURN-NODE-INTEGRATION.md`](Docs/SATURN-NODE-INTEGRATION.md).

Saturn-Node wraps this library behind a private, workload-authenticated service boundary. A compute credential constrains deployment, node, model, context/output limits, concurrency, budget, and expiry.

The library never:

- parses workload credentials;
- opens a network listener;
- decides user authorization or policy;
- loads a model outside the service allowlist;
- receives Apple Container control commands;
- executes agent tools.

## Relationship to Saturn

- **User client:** Saturn One
- **Advanced operator:** Saturn Container
- **Control plane and container execution authority:** Saturn-Control
- **Container runtime adapter:** Saturn-Control Container Runner
- **Agent runtime:** agent logic inside an Apple Container workload
- **Inference service:** Saturn-Node
- **Inference library:** this package
- **Governance:** EvoEthics

## Immediate next work

1. Create and approve the separate `saturn-node` service repository.
2. Define the private workload-authenticated inference contract there.
3. Pin one runtime and model manifest.
4. Run real-hardware smoke, cancellation, timeout, disconnect, and restart tests.
5. Validate an agent container using a short-lived compute credential.
6. Validate complete correlation through Saturn-Control before resuming distributed graph work.

System-wide order and acceptance thresholds live in `saturn-control/docs/SATURN-MVP.md`.

## Long-term research direction

After the single-node gate is reliable:

- hardened verifier/drafter acceptance and rollback;
- cross-device Device Graph and Mesh Expert Graph;
- Runtime DAG modeling for routers, embedders, vision, and rerankers;
- telemetry-driven graph scheduling and residency-aware placement;
- expanded episodic-memory, hybrid RAG, benchmarks, and hardware smokes.

Long-term work must remain separable from the stable Saturn-Node inference contract.

## License and copyright

Copyright © 2026 EvoCortexAI S.L. All rights reserved.
