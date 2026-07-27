# saturn-mlx-mesh Repository Guidelines

## Purpose

This repository is the MLX-native inference library loaded inside Saturn-Node.

It is not Saturn-Node, Saturn-Control, an agent runtime, a container orchestrator, or a user-facing product.

## Canonical boundary

```text
Agent container
    -> workload-authenticated Saturn-Node
    -> saturn-mlx-mesh
    -> MLX
```

Saturn-Control runs agent containers through its Container Runner and assigns compute. Saturn-Node validates workload credentials and exposes private inference. This package performs in-process MLX work only.

Read:

- `README.md`
- `Docs/SATURN-NODE-INTEGRATION.md`
- canonical Saturn architecture in the Saturn-Control repository.

## Owned responsibilities

- model loading;
- MLX generation;
- token streaming primitives;
- cancellation hooks;
- KV-cache behavior;
- placement and graph logic;
- inference telemetry primitives;
- deterministic simulation and hardware smoke support.

## Prohibited scope

Do not add:

- Apple Container lifecycle code;
- CLI/container process execution;
- network listeners or HTTP routes;
- workload token parsing;
- user authentication;
- Saturn-Control policy or orchestration;
- agent tools or task loops;
- frontend models;
- service installation or launchd assets;
- production secrets or endpoint details.

A separate `saturn-node` repository owns the private service.

## MVP priority

Prioritize one pinned, repeatable, cancellable MLX inference path on target Apple Silicon.

Do not expand distributed mesh, Runtime DAG, automatic scheduling, speculative research, or episodic-memory product behavior while model load, streaming, cancellation, timeout, disconnect, and restart gates are unreliable.

## Code rules

- Preserve Swift actor isolation and `Sendable` correctness.
- Keep public APIs small and source-compatible unless a reviewed breaking change is required.
- Use structured concurrency.
- Ensure every stream finishes exactly once.
- Propagate cancellation promptly.
- Bound memory, token generation, and queues.
- Keep standard telemetry metadata-only.
- Keep unit tests deterministic and free of model downloads.
- Put real-hardware work in opt-in smoke targets.
- Pin production runtime revisions in Saturn-Node, not by pretending SwiftPM ranges are deployment manifests.

## Change discipline

Before editing:

```sh
git status --short --branch
git diff --name-only
```

Keep task diffs narrow. Preserve unrelated work. Ask before architecture changes, dependency changes, breaking public API changes, target/platform changes, commits, pushes, releases, or production-hardware operations.

After editing:

```sh
swift build
swift test
git diff --check
git status --short
```

If real MLX verification is required, report target hardware, model, dependency revisions, command, timing, and outcome without prompt or generated content.
