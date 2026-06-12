// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// MeshSession.swift
// Factory and session owner for Mesh LLM models.
//
// This is the public entry point matching the exact API:
//
//   let mesh = MeshSession(controlPlane: .local, policy: .appleSiliconBalanced)
//   let model = try await mesh.loadModel(id: "...", role: .primary)
//   let stream = try await model.generate(prompt: "...", ..., speculativeGamma: 4)
//
// v0.1: Single-node MLX, KV cache reuse, simplified speculative with proper
// rejection fallback, placement policy wired at load, actor isolation,
// telemetry hooks. See README and Docs/.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

// The following two imports are required for the macro expansions
// produced by #hubDownloader() and #huggingFaceTokenizerLoader()
// (they come from the packages added in Package.swift for this real-loading pass).
import HuggingFace
import Tokenizers

public enum ControlPlane: Sendable {
    case local          // running directly on a Saturn-Node (most common for v0.1)
    case remote(String) // future: control plane address for multi-node coordination
}

public enum SessionPolicy: Sendable {
    case appleSiliconBalanced
    // Future: powerFirst, latencyFirst, accuracyFirst, etc.
}

/// MeshSession is the main entry point (actor for safe concurrent use from multiple tasks).
public actor MeshSession {
    public let controlPlane: ControlPlane
    private let policy: any PlacementPolicy
    private let telemetry = MeshTelemetry()
    private let defaultSpeculativeGamma: Int?

    public init(
        controlPlane: ControlPlane = .local,
        policy: SessionPolicy = .appleSiliconBalanced,
        defaultSpeculativeGamma: Int? = nil
    ) {
        self.controlPlane = controlPlane
        self.defaultSpeculativeGamma = defaultSpeculativeGamma

        switch policy {
        case .appleSiliconBalanced:
            self.policy = PlacementPolicyKind.appleSiliconBalanced
        }
    }

    /// Load a model (and optionally a drafter for speculative decoding).
    public func loadModel(
        id: String,
        role: ModelRole = .primary,
        drafterId: String? = nil
    ) async throws -> MeshModel {
        let decision = policy.decide(role: role)
        let loadStart = Date()

        // Real loader (wired in the narrow real-loading first pass).
        // Uses the exact macro form recommended by mlx-swift-lm and already
        // documented in the previous skeleton stub.
        let factory = LLMModelFactory.shared
        let configuration = ModelConfiguration(id: id)

        let container = try await factory.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration
        )

        var drafterContainer: ModelContainer?
        if let drafterId {
            let drafterConfig = ModelConfiguration(id: drafterId)
            drafterContainer = try await factory.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: drafterConfig
            )
        }

        let loadDuration = Date().timeIntervalSince(loadStart)

        let model = MeshModel(
            id: id,
            role: role,
            placement: decision,
            telemetry: telemetry,
            speculativeGammaDefault: defaultSpeculativeGamma
        )
        await model.attachContainer(container, drafter: drafterContainer)

        await telemetry.recordLoad(
            modelID: id,
            role: role,
            decision: decision,
            loadDuration: loadDuration
        )

        return model
    }

    public func telemetrySnapshot() async -> TelemetrySnapshot {
        await telemetry.snapshot()
    }
}

// Convenience re-exports so callers can use the types without extra imports.
public typealias SaturnMeshSession = MeshSession
public typealias SaturnMeshModel = MeshModel
