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
// It exercises the narrow real-loading checkpoint:
//   let mesh = MeshSession(...)
//   let model = try await mesh.loadModel(id: "mlx-community/Qwen3-8B-4bit")
//   let stream = try await model.generate(prompt: "...", maxTokens: 32)
//
// Expected: real decoded tokens are streamed from a live MLX container.

import Foundation
import SaturnMLXMesh

@main
struct SaturnMLXMeshSmoke {
    static func main() async throws {
        print("=== saturn-mlx-mesh real-loading smoke ===")
        print("Control plane: .local")
        print("Policy: .appleSiliconBalanced")
        print("Model: mlx-community/Qwen3-8B-4bit (or the 4-bit Qwen variant available)")
        print("Prompt: \"Explain Saturn mesh inference in one short sentence.\"")
        print("maxTokens: 32")
        print("")

        let mesh = MeshSession(
            controlPlane: .local,
            policy: .appleSiliconBalanced
        )

        let model = try await mesh.loadModel(
            id: "mlx-community/Qwen3-8B-4bit",
            role: .primary
        )

        let stream = try await model.generate(
            prompt: "Explain Saturn mesh inference in one short sentence.",
            maxTokens: 32,
            temperature: 0.7
        )

        print("Tokens: ", terminator: "")
        for try await token in stream {
            print(token.text, terminator: "")
            fflush(stdout)
        }
        print("\n")
        print("Smoke completed successfully (real tokens received).")
    }
}
