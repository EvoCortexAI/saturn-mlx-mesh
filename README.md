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

- `MeshSession` — factory, control plane selection, policy
- `MeshModel` — `@MainActor`, generation (standard + speculative path), reusable KV cache hooks
- `PlacementPolicy` + `AppleSiliconBalanced` — `MeshExecutionUnit`, `PlacementDecision`, `decide(role:)`
- `MeshTelemetry` — actor for load + generation records (tokens/s, speculative acceptance, etc.)

## Status (v0.1)

- ✅ Exact public API surface
- ✅ Placement policy wired at load time + recorded in telemetry
- ✅ Actor isolation where appropriate
- ✅ Telemetry actor + snapshot
- ✅ Basic unit tests (policy, telemetry, session wiring)
- ⚠️ Model loading is currently stubbed (see `MeshSession.loadModel`). The real loader requires the current `mlx-swift-lm` Hub downloader + tokenizer loader macros + the corresponding products (`MLXHuggingFace` etc.). Replace the stub and add products when integrating on target Saturn-Node hardware.
- Speculative decoding: public API + telemetry support present. Full drafter + verifier acceptance loop (with proper rejection math) is intentionally minimal/stubbed pending basic end-to-end streaming + cache validation.

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

## Next (post v0.1)

- Restore production loader + add `MLXHuggingFace` + tokenizer products.
- Real-hardware streaming + speculative validation (with measured acceptance rates).
- Expose richer cache control and `maxKVSize` application.
- Dynamic placement using live `TelemetrySnapshot`.
- Multi-node / mesh routing (when control plane starts coordinating multiple Saturn-Nodes).

## License / Copyright

Copyright © 2026 EvoCortexAI S.L. All rights reserved.

---

This repository is intentionally separate from `saturn-control`. It is the MLX-native mesh execution component for private on-device inference in the EvoIntelligenceFabric.