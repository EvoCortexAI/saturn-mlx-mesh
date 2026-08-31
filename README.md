# saturn-mlx-mesh

In-process MLX library loaded by Saturn-Node. Not a service. No listener, credentials, or orchestration.

**License:** Apache License 2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE))  
**Tag:** `0.2.0` @ `9aab96a2e24817fbb1898f8c133ad44469986805`

Diagrams: [`Docs/ARCHITECTURE.md`](Docs/ARCHITECTURE.md). Node seam: [`Docs/SATURN-NODE-INTEGRATION.md`](Docs/SATURN-NODE-INTEGRATION.md).

## Baseline

Swift tools **6.3**, Xcode **26.6**, macOS/iOS **26**. Ordinary CI is weight-free. Real MLX is opt-in Apple Silicon smoke only.

## Node contract (`0.2.x`)

Saturn-Node depends on `MLXInferenceRuntime`: `capabilities()`, `generate(_:)`, `cancel(requestID:)`.

| Type | Use |
|------|-----|
| `SimulatedMLXInferenceRuntime` | CI. No weights. |
| `MeshModelInferenceRuntime` | Real MLX. Opt-in smoke / hardware gate. |
| `AcceptanceModelPin.primaryModelID` | `mlx-community/Qwen3-8B-4bit` |

Graph, placement DAG, speculative, and episodic research stay in-tree and **out of** the Node adapter contract.

```sh
swift run SaturnMLXMeshSmoke
swift run SaturnMLXMeshSmoke --cancel-recovery
```

Metadata-only. Library smoke does not prove Node auth, transport, or SN01.

## Owns / does not own

Owns: load, stream, native cancel, KV/resource hooks, sim + smoke, telemetry primitives.

Does not own: Control, Container Runner, agent tools, workload credentials, authority receipts, network listener, frontend, EvoEthics policy admin.

## Verify

```sh
swift package dump-package >/dev/null
swift build
swift test
```

Node consumes this package with `.upToNextMinor(from: "0.2.0")`. A floating `branch: "main"` pin is not a release contract.
