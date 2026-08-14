# saturn-mlx-mesh Repository Guidelines

## Purpose

This repository is the MLX-native inference library loaded inside Saturn-Node.

It is not Saturn-Node, Saturn-Control, an agent runtime, a container orchestrator, or a user-facing product.

## Toolchain baseline

- Swift tools: **6.3**
- CI Xcode: **26.6**
- Package deployment floors: **macOS 26** and **iOS 26**
- Real-hardware MVP acceptance target: Apple Silicon macOS host used by Saturn-Node

Do not lower the package deployment floors or Swift tools version without an explicit compatibility decision. Normal CI remains deterministic and must not download model weights.

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
- `Docs/ACCEPTANCE-MODEL.md`
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

## Hardware acceptance

The opt-in executable exercises the same `MLXInferenceRuntime` contract consumed by Saturn-Node:

```sh
swift run SaturnMLXMeshSmoke
swift run SaturnMLXMeshSmoke --cancel-recovery
```

The first command proves real model load plus streamed completion and records timing metadata. The second additionally proves explicit cancellation and a successful subsequent request on the same loaded runtime. Generated content is suppressed unless `--show-content` is explicitly supplied.

A passing smoke is necessary but not sufficient to close hardware acceptance. Record host/toolchain/dependency/model revisions and repeat from a fresh process as described in `Docs/ACCEPTANCE-MODEL.md`. Managed Saturn-Node restart and authenticated service behavior remain Node-owned gates.

## Change discipline

Before editing:

```sh
git status --short --branch
git diff --name-only
```

Keep task diffs narrow. Preserve unrelated work. Ask before architecture changes, dependency changes, breaking public API changes, target/platform changes, commits, pushes, releases, or production-hardware operations.

After editing:

```sh
swift package dump-package >/dev/null
swift build
swift test
git diff --check
git status --short
```

If real MLX verification is required, report target hardware, model, dependency revisions, command, timing, and outcome without prompt or generated content.
