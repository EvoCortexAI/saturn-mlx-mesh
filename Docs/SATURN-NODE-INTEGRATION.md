# Saturn-Node Integration Boundary

**Status:** Proposed service integration; no service implementation exists in this repository

## Decision

`saturn-mlx-mesh` is an in-process MLX inference library. A separate `saturn-node` service owns workload authentication, network transport, model allowlisting, quotas, service lifecycle, and operational recovery.

```text
Agent container
    -> private TLS + short-lived compute credential
    -> Saturn-Node service
    -> SaturnMLXMesh library
    -> MLX
```

Saturn-Control issues the compute assignment and credential. The agent container does not select arbitrary models or load model weights directly.

## Service-to-library adapter

The future Saturn-Node service should depend on a narrow internal protocol:

```swift
protocol MLXInferenceRuntime: Sendable {
    func capabilities() async throws -> InferenceCapabilities

    func generate(
        _ request: ValidatedInferenceRequest
    ) -> AsyncThrowingStream<InferenceChunk, Error>

    func cancel(requestID: InferenceRequestID) async
}
```

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

The exact token format and private wire endpoint belong to the future `saturn-node` repository. This library must not define or parse them.

## Cancellation

Cancellation must propagate:

```text
Saturn One / Saturn Container stop
    -> Saturn-Control
    -> agent deployment or agent session
    -> Saturn-Node request
    -> SaturnMLXMesh generation task
```

Requirements:

- cancellation closes the stream once;
- no orphan generation continues;
- partial output is not misreported as success;
- terminal metadata identifies cancellation without prompt/response content;
- resource cleanup is bounded and testable.

## Telemetry

The library may emit:

- model and runtime identifiers;
- request correlation ID;
- load and generation timing;
- token counts;
- memory/capacity indicators;
- cancellation and terminal outcome;
- placement and execution-unit metadata.

It must not emit prompt or generated-response content through standard telemetry.

Saturn-Node maps library telemetry into workload-scoped usage evidence. Saturn-Control owns cross-component audit correlation.

## Failure mapping

Library failures should be typed so Saturn-Node can distinguish:

- model unavailable;
- model load failure;
- unsupported generation parameter;
- context limit exceeded;
- capacity exhausted;
- generation timeout;
- cancellation;
- malformed internal output;
- runtime incompatibility.

Authentication, authorization, lease expiry, revocation, and policy denial are Saturn-Node or Saturn-Control failures, not library failures.

## Test boundary

This repository tests:

- deterministic simulated generation;
- stream completion;
- cancellation behavior exposed by the library;
- model-load and generation errors;
- telemetry correctness;
- hardware smoke behavior.

The future Saturn-Node repository tests:

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
- automatic fleet scheduling.
