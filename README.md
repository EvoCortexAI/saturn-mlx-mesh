# saturn-mlx-mesh

**Mesh LLM extension for MLX on Apple Silicon**

Part of [EvoIntelligenceFabric](https://evocortex.ai) — the execution plane (Saturn-Node).

v0.1 — initial, verification-first, single-node implementation.

## Public API (exact contract)

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

## Key Components

- `MeshSession` — actor factory, control plane selection, policy
- `MeshModel` — actor-isolated, generation (standard + speculativeGamma surface), telemetry hooks, prepared for future KV reuse
- `PlacementPolicy` + `AppleSiliconBalanced` — `MeshExecutionUnit`, `PlacementDecision`, `decide(role:)` (with weighted cost notes)
- `MeshTelemetry` — actor for load + generation records (truthful speculative fields)

## Status (v0.1 — skeleton-hardening phase)

- ✅ Exact public API surface (MeshSession + MeshModel are actors)
- ✅ Placement policy wired + recorded in telemetry
- ✅ Actor isolation
- ✅ Telemetry (truthful: speculative fields only when drafter was attached *and* used)
- ✅ Stream completion fixed (finish() called on success)
- ✅ Basic unit tests + new coverage for generate behavior (.notLoaded, stream finishes, no-spec telemetry when no drafter)
- ⚠️ Model loading (`loadModel`) is stubbed. Real LLMModelFactory + Hub downloader / tokenizer wiring is the next substantial milestone.
- ⚠️ KV cache reuse: comments + newCache prep exist, but current path uses high-level generate (full TokenIterator reuse is TODO).
- Speculative: API + telemetry support present. Currently falls back to normal generation. Full drafter + acceptance/rejection is TODO (see code).

See [Docs/mesh-llm-mlx-extension.md](Docs/mesh-llm-mlx-extension.md) for the full vision, math references (placement cost model, speculative speedup formula `1 + γ·α`), and rollout notes.

## Build & Test

```bash
swift build
swift test
```

Requires Apple Silicon (macOS 15+ / iOS 18+ target) for real MLX execution. The unit tests are designed to pass without downloading large models.

## Dependencies

- `mlx-swift-lm` (MLXLLM + MLXLMCommon)

## Relationship to the Rest of Saturn

- **Control plane**: `saturn-control` (orchestration, node registry, scheduling, OpenAI-compatible gateway). Never runs heavy inference.
- **Execution plane**: This package + surrounding Saturn-Node runtime on Apple Silicon devices (primary/secondary/drafter roles, UMA placement, power telemetry).
- **Client plane**: Saturn One (Apple app).

Nodes register capabilities with labels such as `["mlx", "primary"]`. This library powers the actual inference work on those nodes.

## Next (recommended order)

1. Real model loading (LLMModelFactory + downloader/tokenizer products) — next substantial milestone.
2. Only after real loading: KV cache reuse + real drafter/verifier speculative.
3. Real-hardware smoke + telemetry back to control plane.
4. Dynamic placement, multi-node, etc.

## License / Copyright

Copyright © 2026 EvoCortexAI S.L. All rights reserved.

---

This repository is intentionally separate from `saturn-control`. It is the MLX-native mesh execution component for private on-device inference in the EvoIntelligenceFabric.