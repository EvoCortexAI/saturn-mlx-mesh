// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// SaturnMLXMeshSmoke
//
// Opt-in hardware smoke test / demo executable.
//
// This target is intentionally **not** depended on by the test target.
// `swift test` (and normal CI) will continue to use the simulation hook
// (_enableTestSuccessSimulation) and will never download model weights.
//
// Run manually on Apple Silicon hardware (after `swift build` or via `swift run`):
//     swift run SaturnMLXMeshSmoke
//
// Model identity comes from `AcceptanceModelPin` (mesh#1 / KF primary).
// Expected: real decoded tokens are streamed from a live MLX container.
// Record load time, TTFT, tokens/sec, cancel+reclaim, and dependency pins
// after a successful run — do not treat a single ad-hoc CLI success as closed.

import Foundation
import SaturnMLXMesh

@main
struct SaturnMLXMeshSmoke {
    static func main() async throws {
        let modelID = AcceptanceModelPin.primaryModelID
        let prompt = AcceptanceModelPin.acceptancePrompt
        let maxTokens = AcceptanceModelPin.smokeMaxTokens

        print("=== saturn-mlx-mesh real-loading smoke ===")
        print("Control plane: .local")
        print("Policy: .appleSiliconBalanced")
        print("Model: \(modelID) (\(AcceptanceModelPin.primaryRole))")
        print("Prompt: \"\(prompt)\"")
        print("maxTokens: \(maxTokens)")
        print("")

        let mesh = MeshSession(
            controlPlane: .local,
            policy: .appleSiliconBalanced
        )

        let model = try await mesh.loadModel(
            id: modelID,
            role: .primary
        )

        let stream = try await model.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: 0.7
        )

        print("Tokens: ", terminator: "")
        for try await token in stream {
            print(token.text, terminator: "")
            fflush(stdout)
        }
        print("\n")
        print("Smoke completed successfully (real tokens received).")
        print("Next: record timings + dependency pins into mesh#1 acceptance notes.")
    }
}
