// swift-tools-version: 6.3
// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// saturn-mlx-mesh
// MLX-native inference library for Apple Silicon (EvoIntelligenceFabric execution plane).
//
// The package baseline intentionally matches the Saturn-Node deployment toolchain.
// Real hardware acceptance is performed separately from deterministic CI.

import PackageDescription

let package = Package(
    name: "saturn-mlx-mesh",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        .library(
            name: "SaturnMLXMesh",
            targets: ["SaturnMLXMesh"]
        )
    ],
    dependencies: [
        // mlx-swift-lm provides MLXLLM + MLXLMCommon + the underlying MLX runtime.
        // The SwiftPM lower bound is a development compatibility range; deployed
        // Saturn-Node acceptance must record the exact resolved revision.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            from: "3.31.0"
        ),
        // Required for the #hubDownloader() / #huggingFaceTokenizerLoader() macros
        // (Downloader + TokenizerLoader implementations + HubClient).
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
                // These provide the modules required by the Hugging Face integration macros
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
        // Opt-in hardware acceptance executable. It is never depended on by the
        // test target, so ordinary `swift test` / CI remains deterministic and
        // does not download model weights.
        .executableTarget(
            name: "SaturnMLXMeshSmoke",
            dependencies: ["SaturnMLXMesh"],
            path: "Sources/SaturnMLXMeshSmoke"
        )
    ]
)
