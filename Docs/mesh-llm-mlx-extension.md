# Saturn MLX Mesh — LLM Extension for MLX on Apple Silicon (v0.1)

**Package**: `saturn-mlx-mesh` (SwiftPM library `SaturnMLXMesh`)  
**Status**: Initial implementation, verification-first, single-node only.  
**Part of**: EvoIntelligenceFabric — Execution plane (Saturn-Node)

## Vision (preserved from Saturn Mesh design)

The Saturn Mesh aims to deliver:

- **UMA-first heterogeneous placement** on Apple Silicon nodes (GPU / unified / CPU / ANE as appropriate per role).
- **Efficient KV cache reuse** across turns and across speculative steps.
- **Speculative decoding** (with drafter models) for latency wins while preserving exact verifier semantics.
- **First-class telemetry** (tokens/s, memory pressure, load times) that can later feed dynamic placement cost models.
- Clean boundaries so the control plane (Saturn-Control) only ever sees high-level task orchestration and node-reported metrics, never raw inference state.

This package is the concrete MLX implementation of the execution side of that vision.

## Current Scope (v0.1 — strictly minimal & verifiable)

- Single Apple Silicon node only. No real cross-node mesh, no layer sharding.
- `MeshSession` + `MeshModel` public surface exactly as specified in the council-reviewed prompt.
- Placement policy is fully wired at load time and decisions are recorded in telemetry. Physical device selection inside MLX remains largely implicit (Metal + UMA) for v0.1.
- KV cache reuse is mandatory and implemented via the low-level `newCache` + `TokenIterator` path recommended by the mlx-swift-lm / WWDC25 material.
- Speculative decoding is present in two forms:
  1. Attempt to use native speculative helpers from MLXLMCommon when available.
  2. Explicit simplified propose/verify fallback with correct rejection sampling (no incorrect tokens are ever emitted to the caller).
- Telemetry actor captures load records and per-generation stats (including speculative acceptance counts).
- Basic error handling and clear `MeshModelError` cases.
- Comments throughout reference the placement cost model and the speculative speedup formula.

**Explicit non-goals for v0.1** (per the prompt):
- No cross-device layer sharding.
- No production-grade multi-node router.
- No full MTP / tree speculative or advanced acceptance sampling.
- Real model loads in unit tests are avoided (they require large downloads and Apple Silicon Metal).

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

## Files in this package (v0.1)

```
saturn-mlx-mesh/
├── Package.swift
├── Sources/SaturnMLXMesh/
│   ├── MeshSession.swift          // factory + session owner
│   ├── MeshModel.swift            // @MainActor model + KV + speculative
│   ├── PlacementPolicy.swift      // MeshExecutionUnit, PlacementDecision, AppleSiliconBalanced
│   ├── MeshTelemetry.swift        // actor for load/generation records
│   ├── MeshExecutionUnit.swift    // small re-export helper
│   └── (minimal internal helpers)
├── Tests/SaturnMLXMeshTests/
│   └── MeshModelTests.swift       // policy, telemetry, session wiring (no heavy loads)
└── Docs/
    └── mesh-llm-mlx-extension.md  // this file
```

## Dependencies

- `mlx-swift-lm` (provides `MLXLLM` + `MLXLMCommon`)
- Targets macOS 15+ / iOS 18+

## Validation performed

After generation the package was exercised with:

```bash
swift build
swift test
```

See the generating agent transcript for concrete output and any required follow-up fixes against the exact `mlx-swift-lm` surface present on the validation machine.

## Next steps (post v0.1)

1. This is the home of the execution-plane mesh package (Saturn-Node). It is intentionally separate from saturn-control (the control plane).
2. Add a small real-hardware smoke that exercises streaming + speculativeGamma + telemetry snapshot (can live in a `SaturnNodeBench` target).
3. Wire node-reported telemetry back into Saturn-Control (via existing heartbeat / task event paths).
4. Evolve placement policy to consume live `TelemetrySnapshot` data for dynamic decisions.
5. When multi-node support arrives, extend `ControlPlane.remote(...)` and add wire contracts.

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
