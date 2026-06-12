// swift-tools-version: 6.0
// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// saturn-mlx-mesh
// Mesh LLM extension for MLX on Apple Silicon (EvoIntelligenceFabric execution plane).
//
// v0.1 initial skeleton.
// Part of the Saturn-Node / execution plane. See README.md and Docs/ for status.

import PackageDescription

let package = Package(
    name: "saturn-mlx-mesh",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        .library(
            name: "SaturnMLXMesh",
            targets: ["SaturnMLXMesh"]
        )
    ],
    dependencies: [
        // mlx-swift-lm provides MLXLLM + MLXLMCommon + the underlying MLX runtime.
        // Use a recent compatible version; update after validating against
        // actual deployed Saturn-Node hardware.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            from: "3.31.0"
        ),
        // Required for the #hubDownloader() / #huggingFaceTokenizerLoader() macros
        // (Downloader + TokenizerLoader implementations + HubClient).
        // Exact versions taken from mlx-swift-lm's own recommended consumer setup.
        .package(
            url: "https://github.com/huggingface/swift-huggingface",
            from: "0.9.0"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.3.0"
        )
    ],
    targets: [
        .target(
            name: "SaturnMLXMesh",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                // These provide the modules required by the HuggingFace integration macros
                // (HuggingFace.HubClient and Tokenizers.Tokenizer + the adaptor).
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            path: "Sources/SaturnMLXMesh"
        ),
        .testTarget(
            name: "SaturnMLXMeshTests",
            dependencies: ["SaturnMLXMesh"],
            path: "Tests/SaturnMLXMeshTests"
        ),
        // Opt-in hardware smoke test / executable.
        // Never depended on by the test target, so `swift test` and normal CI
        // continue to use only the _enableTestSuccessSimulation hook (no model downloads).
        // Run manually on Apple Silicon with: swift run SaturnMLXMeshSmoke
        .executableTarget(
            name: "SaturnMLXMeshSmoke",
            dependencies: ["SaturnMLXMesh"],
            path: "Sources/SaturnMLXMeshSmoke"
        )
    ]
)
