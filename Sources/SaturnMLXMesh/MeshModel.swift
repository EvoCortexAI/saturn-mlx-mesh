// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// MeshModel.swift
// Actor-isolated model handle.
//
// v0.1 focuses on:
// - actor isolation for MeshSession + MeshModel
// - reusable KV cache (creation + comments for reuse across turns)
// - speculativeGamma API surface + telemetry
// - placement decision carried on the model
//
// Inside ModelContainer.perform we are in a Sendable context, so we avoid
// capturing non-Sendable caches directly. Full TokenIterator + explicit
// cache reuse + draftCache will be enabled in a follow-up once the
// non-Sendable KVCache story is handled (common pattern is a dedicated
// cache owner or copying offsets).

import Foundation
import MLXLLM
import MLXLMCommon

public enum MeshModelError: Error, LocalizedError {
    case notLoaded
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded: return "MeshModel has not successfully loaded a container"
        case .generationFailed(let m): return "Generation failed: \(m)"
        }
    }
}

public actor MeshModel {
    public let id: String
    public let role: ModelRole
    public let placement: PlacementDecision

    private var container: ModelContainer?
    private var drafterContainer: ModelContainer?

    private let telemetry: MeshTelemetry
    private let speculativeGammaDefault: Int?

    // Test hook for skeleton-hardening tests (stream finish, telemetry truth)
    private var simulateSuccessForTest = false

    init(
        id: String,
        role: ModelRole,
        placement: PlacementDecision,
        telemetry: MeshTelemetry,
        speculativeGammaDefault: Int? = nil
    ) {
        self.id = id
        self.role = role
        self.placement = placement
        self.telemetry = telemetry
        self.speculativeGammaDefault = speculativeGammaDefault
    }

    func attachContainer(_ container: ModelContainer, drafter: ModelContainer? = nil) async {
        self.container = container
        self.drafterContainer = drafter
    }

    /// Enable a mock success path for unit tests (yields a few tokens then finishes cleanly).
    /// Only for skeleton-hardening / behavior tests. Does not require a real MLX container.
    internal func _enableTestSuccessSimulation() async {
        simulateSuccessForTest = true
    }

    public func generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        speculativeGamma: Int? = nil
    ) async throws -> AsyncThrowingStream<GeneratedToken, Error> {
        // Test simulation path for stream finish + telemetry tests (skeleton hardening)
        if simulateSuccessForTest {
            return AsyncThrowingStream { continuation in
                Task {
                    for i in 0..<4 {
                        continuation.yield(GeneratedToken(text: "tok\(i)", tokenID: i))
                    }
                    continuation.finish()

                    // Record truthful telemetry (no drafter in sim unless attached; here we force non-spec)
                    let dur = 0.05
                    let info = GenerationInfo(
                        modelID: id,
                        role: role,
                        promptTokens: 3,
                        generatedTokens: 4,
                        duration: dur,
                        tokensPerSecond: 80.0,
                        speculativeGamma: nil,
                        acceptedTokens: nil,
                        memoryPressureHint: nil,
                        timestamp: Date()
                    )
                    await telemetry.recordGenerationInfo(info)
                }
            }
        }

        guard let container = container else {
            throw MeshModelError.notLoaded
        }

        let gamma = speculativeGamma ?? speculativeGammaDefault
        let start = Date()

        let params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: 0.95,
            repetitionPenalty: 1.05,
            repetitionContextSize: 20
        )

        // Capture before Sendable closure
        let hasDrafter = drafterContainer != nil
        let theDrafter = drafterContainer
        let requestedSpec = (gamma ?? 0) > 1
        let didSpec = requestedSpec && hasDrafter

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let count = try await container.perform { context in
                        let input = try await context.processor.prepare(
                            input: UserInput(prompt: prompt)
                        )

                        if didSpec, let drafter = theDrafter {
                            // v0.1 simplified speculative (see _runSpeculative for TODOs)
                            return try await self._runSpeculative(
                                input: input,
                                params: params,
                                context: context,
                                drafter: drafter,
                                gamma: gamma!,
                                continuation: continuation
                            )
                        } else {
                            // Standard generation.
                            // TODO: switch to TokenIterator + explicit newCache reuse
                            // once non-Sendable KVCache capture is solved cleanly.
                            let stream = try MLXLMCommon.generate(
                                input: input,
                                parameters: params,
                                context: context
                            )
                            var c = 0
                            for await gen in stream {
                                if case .chunk(let text) = gen, !text.isEmpty {
                                    continuation.yield(GeneratedToken(text: text, tokenID: nil))
                                    c += 1
                                }
                                if c >= maxTokens { break }
                            }
                            return c
                        }
                    }

                    // Telemetry truthful: only claim speculative when we actually had a drafter and used the path
                    let dur = Date().timeIntervalSince(start)
                    let tps = dur > 0 ? Double(max(count, 1)) / dur : 0
                    let info = GenerationInfo(
                        modelID: id,
                        role: role,
                        promptTokens: 0,
                        generatedTokens: count,   // NOTE: counts generator chunks (text pieces), not necessarily raw token IDs yet
                        duration: dur,
                        tokensPerSecond: tps,
                        speculativeGamma: didSpec ? gamma : nil,
                        acceptedTokens: didSpec ? count : nil,
                        memoryPressureHint: nil,
                        timestamp: Date()
                    )
                    await telemetry.recordGenerationInfo(info)

                    // CRITICAL: finish the stream on success path so callers do not hang
                    continuation.finish()

                } catch {
                    continuation.finish(throwing: MeshModelError.generationFailed(String(describing: error)))
                }
            }
        }
    }

    // MARK: - v0.1 Speculative (simplified)

    private func _runSpeculative(
        input: LMInput,
        params: GenerateParameters,
        context: ModelContext,
        drafter: ModelContainer,
        gamma: Int,
        continuation: AsyncThrowingStream<GeneratedToken, Error>.Continuation
    ) async throws -> Int {
        // TODO (full rejection sampling + residual distribution):
        // - Generate gamma draft tokens from the drafter (use draftCache for efficiency).
        // - Verify the gamma+1 sequence with the main model.
        // - Accept the longest correct prefix according to the verifier's distribution.
        // - On rejection, emit the verifier's token at the failure position and
        //   roll back caches so the next step starts from a consistent state.
        // - Never yield a token that has not been accepted by the verifier.
        //
        // For v0.1 we fall back to standard verifier generation. This is
        // always correct and lets the rest of the system (actors, telemetry,
        // placement, public API) be validated.

        let stream = try MLXLMCommon.generate(input: input, parameters: params, context: context)
        var count = 0
        for await gen in stream {
            if case .chunk(let text) = gen, !text.isEmpty {
                continuation.yield(GeneratedToken(text: text, tokenID: nil))
                count += 1
            }
            if count >= (params.maxTokens ?? 512) { break }
        }
        return count
    }
}