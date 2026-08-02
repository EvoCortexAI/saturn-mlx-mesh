# Saturn-Node Integration Boundary

**Status:** Stable library adapter surface defined; service implementation lives in `saturn-node`

## Decision

`saturn-mlx-mesh` is an in-process MLX inference library. A separate `saturn-node` service owns workload authentication, network transport, model allowlisting, quotas, service lifecycle, and operational recovery.

```text
Agent container
    -> private TLS + short-lived compute credential
    -> Saturn-Node service
    -> SaturnMLXMesh library (MLXInferenceRuntime adapter)
    -> MLX
```

Saturn-Control issues the compute assignment and credential. The agent container does not select arbitrary models or load model weights directly.

## Stable service-to-library adapter

The Saturn-Node service depends on this narrow library protocol (see `SaturnNodeAdapter.swift`):

```swift
public protocol MLXInferenceRuntime: Sendable {
    func capabilities() async throws -> InferenceCapabilities

    func generate(
        _ request: ValidatedInferenceRequest
    ) -> AsyncThrowingStream<InferenceChunk, Error>

    func cancel(requestID: InferenceRequestID) async
}
```

Supporting types:

- `InferenceRequestID` — opaque correlation ID from the caller
- `ValidatedInferenceRequest` — model ID, prompt, max output tokens, optional temperature
- `InferenceCapabilities` / `InferenceModelCapability` / `InferenceRuntimeState`
- `InferenceChunk` — `.started` | `.delta` | `.completed` | `.cancelled`
- `InferenceFinishReason` — `.stop` | `.length` | `.cancelled`
- `MeshInferenceError` — modelUnavailable, capacityExhausted, requestTimeout, cancelled, notLoaded, generationFailed, runtimeUnavailable

The service validates identity, authorization, model, limits, and budget before constructing `ValidatedInferenceRequest`.

The library receives:

- validated model identifier;
- prompt/input required for inference;
- context and output-token bounds;
- approved generation parameters;
- cancellation context;
- metadata correlation identifier.

The library does not receive:

- raw bearer credentials;
- user authentication tokens;
- policy signing keys;
- Container Runner commands;
- Apple Container sockets;
- agent tool definitions;
- database credentials;
- unrestricted model paths.

### Simulation implementation

`SimulatedMLXInferenceRuntime` is the deterministic, CI-safe implementation. It proves completion, cancellation, timeout, capacity, model-unavailable, internal failure, cleanup, and subsequent-request recovery without downloading weights or touching SN01 hardware.

A production implementation that drives a loaded `MeshModel` may be added later behind the same protocol. Real-hardware acceptance remains gated by issue #1.

## Compute credential boundary

Saturn-Node validates a short-lived credential before calling the library. Its effective scope includes:

- workload identity;
- agent deployment ID;
- Saturn-Node ID;
- allowed model;
- context and output limits;
- concurrency;
- request/token budget;
- issue and expiry times;
- revocation/epoch state;
- policy or approval reference when applicable.

The exact token format and private wire endpoint belong to the `saturn-node` repository. This library must not define or parse them.

## Cancellation

Cancellation must propagate:

```text
Saturn One / Saturn Container stop
    -> Saturn-Control
    -> agent deployment or agent session
    -> Saturn-Node request
    -> SaturnMLXMesh generation task
```

Requirements (enforced by the adapter tests):

- cancellation closes the stream once;
- no orphan generation continues;
- partial output is not misreported as success;
- terminal metadata identifies cancellation without prompt/response content;
- resource cleanup is bounded and testable;
- `cancel(requestID:)` is idempotent;
- a subsequent request succeeds after cancellation and after failure.

## Telemetry

The library may emit metadata-only records (`AdapterTelemetryRecord`):

- model and request correlation ID;
- load and generation timing;
- token counts;
- cancellation and terminal outcome.

It must not emit prompt or generated-response content through standard telemetry.

Saturn-Node maps library telemetry into workload-scoped usage evidence. Saturn-Control owns cross-component audit correlation.

## Failure mapping

Library failures are typed so Saturn-Node can distinguish:

- model unavailable;
- model load failure / not loaded;
- capacity exhausted;
- generation timeout;
- cancellation;
- internal generation failure;
- runtime unavailable.

Authentication, authorization, lease expiry, revocation, and policy denial are Saturn-Node or Saturn-Control failures, not library failures.

## Internal APIs that remain non-contractual

The following types and surfaces are **not** part of the stable Saturn-Node adapter contract. Saturn-Node must not depend on them for the MVP inference path:

- `MeshComputationGraph`, `MeshStage`, `RuntimeDAG`, `MeshGraphExecutor`
- `PlacementPolicy`, `PlacementEngine`, `PlacementDecision`, `MeshExecutionUnit`
- `EpisodicMemoryIndex`, `KVCacheManager`, `LayerBudgetAllocator`, `EpisodeKVCache`
- speculative-decoding internals (drafter propose/verify paths inside `MeshModel`)
- `MeshSession.loadModel`, `MeshSession.generateWithMemory`, graph rewrite hooks
- `_enableTestSuccessSimulation`, `_registerModelForGraphTest`, and other test-only helpers

These remain available for research and future single-node hardening after the real-hardware acceptance gate (issue #1) is green.

## Test boundary

This repository tests:

- deterministic simulated generation via `SimulatedMLXInferenceRuntime`;
- stream completion and exactly one terminal outcome;
- idempotent cancellation and consumer-termination cleanup;
- timeout, model-unavailable, capacity, and internal-failure injection;
- subsequent-request recovery after cancel and failure;
- metadata-only telemetry;
- existing MeshSession / placement / graph unit tests (non-contractual).

The `saturn-node` repository tests:

- credential validation;
- model allowlisting;
- quotas and concurrency;
- network streaming;
- disconnect cancellation;
- service restart and rollback;
- workload isolation;
- end-to-end agent-container requests.

## Explicit non-goals

- network server implementation;
- workload token parsing;
- Apple Container integration;
- agent orchestration;
- tool execution;
- user approval;
- public API compatibility;
- automatic fleet scheduling;
- real-hardware model pinning in this PR (see issue #1).
