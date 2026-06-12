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
// HubClient.default and TokenizersLoader() come from MLXHuggingFace (part of
// the mlx-swift-lm package).

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

        // v0.1 loader note:
        // Replace this entire block with a working call to
        // LLMModelFactory.shared.loadContainer(from: #hubDownloader(), using: #huggingFaceTokenizerLoader(), ...)
        // once the Tokenizers / swift-transformers product is added to Package.swift.
        //
        // For the skeleton we still create the actor-isolated MeshModel so that
        // placement policy, telemetry recording, KV cache logic, TokenIterator usage,
        // and the speculative path (with TODOs) can be validated without a real model download.
        let loadDuration = Date().timeIntervalSince(loadStart)

        let model = MeshModel(
            id: id,
            role: role,
            placement: decision,
            telemetry: telemetry,
            speculativeGammaDefault: defaultSpeculativeGamma
        )
        // No real container in skeleton mode — generate() will surface a clear error.
        // Real loader would do: await model.attachContainer(container, drafter: drafterContainer)

        await telemetry.recordLoad(
            modelID: id,
            role: role,
            decision: decision,
            loadDuration: loadDuration
        )

        // For skeleton purposes we return the model. Callers that want real inference
        // must implement the loader (see Docs/).
        return model
    }

    public func telemetrySnapshot() async -> TelemetrySnapshot {
        await telemetry.snapshot()
    }
}

// Convenience re-exports so callers can use the types without extra imports.
public typealias SaturnMeshSession = MeshSession
public typealias SaturnMeshModel = MeshModel
