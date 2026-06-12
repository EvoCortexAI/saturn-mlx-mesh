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
- `MeshSession` + `MeshModel` are true actors. Public API surface matches the design (including speculativeGamma and drafterId).
- Placement policy is wired at load time and decisions recorded in telemetry.
- Model loading (`loadModel`) is wired through `LLMModelFactory` with the Hub downloader and Hugging Face tokenizer loader macros. Real inference still requires Apple Silicon plus local/network access to model weights.
- Generation uses `TokenIterator` with a persistent cache box for the main model path.
- Speculative path (speculativeGamma) is wired in the API and telemetry. When a drafter is attached, the branch runs the loaded drafter model for proposal tokens. Full verifier acceptance/rejection with proper math and cache rollback is still TODO.
- Telemetry is truthful: speculativeGamma/acceptedTokens are only set when a drafter was attached *and* the speculative branch was taken.
- Stream completion is now correct (continuation.finish() called on success paths).
- Basic error handling (.notLoaded) and focused unit tests for stream/ failure / telemetry truthfulness.
- Comments reference the placement cost model and speculative speedup formula.

**Explicit non-goals / current limitations:**
- No cross-device layer sharding or production multi-node.
- No full speculative decoding acceleration yet; the drafter path emits proposal tokens but does not yet verify/accept/reject with the main model.
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

- `mlx-swift-lm` (provides `MLXLLM`, `MLXLMCommon`, and `MLXHuggingFace`)
- `swift-huggingface` (macro support for Hub downloader wiring)
- `swift-transformers` / `Tokenizers` (macro support for Hugging Face tokenizer loading)
- Targets macOS 15+ / iOS 18+

## Validation performed

```bash
swift build
swift test
```

(Includes tests for .notLoaded, stream completion on success, truthful absence of speculative telemetry when no drafter is attached, graph cost modeling, and lightweight session wiring.)

Real model execution is covered by the opt-in `SaturnMLXMeshSmoke` executable rather than normal unit tests, so CI does not download large weights.

## Recommended next steps (per current analysis)

1. Implement full verifier/drafter speculative acceptance: propose gamma tokens from the drafter, verify with the main model, accept/reject with correct sampling, and roll back caches when needed.
2. Expand real-hardware smoke and benchmark coverage for load, generation, KV reuse, drafter path, and telemetry.
3. Wire telemetry back to the control plane and evolve placement using live costs.
4. Add runtime DAG scheduling over the weighted Saturn Mesh graph.
5. Multi-node / remote control-plane execution later.

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
