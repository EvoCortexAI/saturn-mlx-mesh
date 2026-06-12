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
        )
    ],
    targets: [
        .target(
            name: "SaturnMLXMesh",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm")
            ],
            path: "Sources/SaturnMLXMesh"
        ),
        .testTarget(
            name: "SaturnMLXMeshTests",
            dependencies: ["SaturnMLXMesh"],
            path: "Tests/SaturnMLXMeshTests"
        )
    ]
)
