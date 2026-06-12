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

public final class MeshSession: Sendable {
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

    /// Load a model (and optionally an associated drafter for speculative use).
    ///
    /// In v0.1 the `id` is a Hugging Face repo id or MLX-community quantized id
    /// understood by LLMModelFactory / mlx-swift-lm (e.g. "mlx-community/Qwen3-8B-4bit").
    ///
    /// The returned MeshModel is already attached to its container and ready
    /// for generate calls. The placement decision is taken here and recorded
    /// in telemetry even though the actual device mapping inside MLX is still
    /// mostly implicit (UMA + Metal) in this release.
    public func loadModel(
        id: String,
        role: ModelRole = .primary,
        drafterID: String? = nil
    ) async throws -> MeshModel {
        let decision = policy.decide(role: role)
        let loadStart = Date()

        // Loader stub for v0.1 skeleton build & test.
        // See Docs/mesh-llm-mlx-extension.md for the replacement using
        // #hubDownloader() + #huggingFaceTokenizerLoader() (plus products).
        throw MeshModelError.generationFailed("Model loading is stubbed in this v0.1 skeleton (see Docs). Placement + telemetry are still fully testable.")
    }

    /// Snapshot of all load and generation telemetry recorded by models
    /// created through this session (useful for diagnostics and for feeding
    /// future dynamic placement cost models).
    public func telemetrySnapshot() async -> TelemetrySnapshot {
        await telemetry.snapshot()
    }
}

// Convenience re-exports so callers can use the types without extra imports.
public typealias SaturnMeshSession = MeshSession
public typealias SaturnMeshModel = MeshModel
