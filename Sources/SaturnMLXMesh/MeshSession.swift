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

    /// Episodic memory index (v0.2 text-level; v0.3+ will tie into compressed KV caches).
    /// Ingest conversation turns to enable long-context retrieval without full history.
    public let episodicMemory: EpisodicMemoryIndex

    public init(
        controlPlane: ControlPlane = .local,
        policy: SessionPolicy = .appleSiliconBalanced,
        defaultSpeculativeGamma: Int? = nil,
        episodicMemory: EpisodicMemoryIndex? = nil
    ) {
        self.controlPlane = controlPlane
        self.defaultSpeculativeGamma = defaultSpeculativeGamma
        self.episodicMemory = episodicMemory ?? EpisodicMemoryIndex()

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

    /// Ingest new conversation turns into the episodic memory index.
    /// Call this after each user/assistant exchange for long-context support.
    public func ingest(turns: [ConversationTurn]) async {
        await episodicMemory.ingest(turns: turns)
    }

    /// Retrieve relevant episode text (v0.2) to augment a prompt.
    /// Later versions will return EpisodeKVCache objects for direct cache priming.
    public func retrieveContext(for query: String, maxEpisodes: Int = 2) async -> String {
        guard let match = await episodicMemory.match(query: query) else { return "" }

        // For v0.2 just concatenate the turns from the best episode(s).
        // In a real system we would also retrieve compressed KV for that episode.
        let turnsText = match.episode.turns.map { "\($0.role): \($0.content)" }.joined(separator: "\n")
        return "Relevant prior context:\n\(turnsText)\n\n"
    }

    /// Convenience: retrieve episodic context for the prompt and generate.
    /// This is the v0.2 text-level integration point. KV-primed version in v0.3+.
    public func generateWithMemory(
        model: MeshModel,
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        speculativeGamma: Int? = nil
    ) async throws -> AsyncThrowingStream<GeneratedToken, Error> {
        let context = await retrieveContext(for: prompt)
        let augmented = context + prompt
        return try await model.generate(
            prompt: augmented,
            maxTokens: maxTokens,
            temperature: temperature,
            speculativeGamma: speculativeGamma
        )
    }
}

// Convenience re-exports so callers can use the types without extra imports.
public typealias SaturnMeshSession = MeshSession
public typealias SaturnMeshModel = MeshModel
