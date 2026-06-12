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

## Current Scope (v0.1 — strictly minimal & verifiable — skeleton hardening phase)

- Single Apple Silicon node only. No real cross-node mesh, no layer sharding.
- `MeshSession` + `MeshModel` are true actors. Public API surface matches the design (including speculativeGamma and drafterId).
- Placement policy is wired at load time and decisions recorded in telemetry.
- Model loading (`loadModel`) is stubbed — it throws a clear error explaining the required Hub downloader + tokenizer wiring. No real LLMModelFactory / MLX container is acquired yet.
- Generation uses the high-level MLXLMCommon.generate path. Comments and some newCache calls exist for future KV cache reuse, but actual cross-turn reuse via TokenIterator is not yet active (TODO).
- Speculative path (speculativeGamma) is wired in the API and telemetry. When no drafter is attached (or in current skeleton), it safely falls back to normal generation. Full drafter proposal + verifier acceptance/rejection with proper math + draftCache rollback is still TODO (see _runSpeculative).
- Telemetry is truthful: speculativeGamma/acceptedTokens are only set when a drafter was attached *and* the speculative branch was taken. generatedTokens currently reflects generator-emitted chunks (not raw token IDs).
- Stream completion is now correct (continuation.finish() called on success paths).
- Basic error handling (.notLoaded) and focused unit tests for stream/ failure / telemetry truthfulness.
- Comments reference the placement cost model and speculative speedup formula.

**Explicit non-goals / current limitations (skeleton phase):**
- No real model loading / tokenizer / downloader (see recommended milestone order).
- No cross-device layer sharding or production multi-node.
- No full speculative decoding acceleration (falls back).
- No KV cache reuse across generations yet.
- Real hardware smoke tests are manual only.

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
│   ├── MeshModel.swift            // actor-isolated model (generation + speculative API + telemetry)
│   ├── PlacementPolicy.swift      // MeshExecutionUnit, PlacementDecision, AppleSiliconBalanced + weighted cost notes
│   ├── MeshTelemetry.swift        // actor for load/generation records (truthful speculative fields)
│   ├── MeshExecutionUnit.swift    // small re-export helper
│   └── (minimal internal helpers)
├── Tests/SaturnMLXMeshTests/
│   └── SaturnMLXMeshTests.swift   // policy, telemetry, session + generate behavior (notLoaded, stream finish, no-spec telemetry)
└── Docs/
    └── mesh-llm-mlx-extension.md  // this file
```

## Dependencies

- `mlx-swift-lm` (provides `MLXLLM` + `MLXLMCommon`)
- Targets macOS 15+ / iOS 18+

## Validation performed

```bash
swift build
swift test
```

(Now includes tests for .notLoaded, stream completion on success, and truthful absence of speculative telemetry when no drafter is attached.)

The loader remains intentionally stubbed per the hardening milestone plan. Real loading is the next major step.

## Recommended next steps (per current analysis)

Follow the skeleton-hardening then real-inference order:

1. (Done in this pass) Stream completion, truthful telemetry, honest docs, focused generate tests.
2. Implement real model loading (LLMModelFactory + proper Hub downloader / #huggingFaceTokenizerLoader() + Tokenizers product). This is the next substantial milestone.
3. Only after real loading works: tackle KV cache reuse (TokenIterator + persistent cache across turns) and real drafter/verifier speculative decoding.
4. Add real-hardware smoke / benchmark target.
5. Wire telemetry back to control plane, evolve placement, etc.
6. Multi-node later.

The package is intentionally separate from saturn-control.

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
