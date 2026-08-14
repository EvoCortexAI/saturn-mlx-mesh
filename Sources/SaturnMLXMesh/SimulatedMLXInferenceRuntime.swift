// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// SimulatedMLXInferenceRuntime.swift
// Deterministic, simulation-only MLXInferenceRuntime for unit tests and CI.
// Never downloads weights. Real path: MeshModelInferenceRuntime.

import Foundation

/// Configuration for deterministic simulation used by unit tests and CI.
public struct SimulatedInferenceConfig: Sendable {
    public var modelID: String
    public var maxContextTokens: Int
    public var maxOutputTokens: Int
    public var isLoaded: Bool
    public var maximumConcurrentRequests: Int
    public var chunkCount: Int
    public var chunkDelayNanoseconds: UInt64
    public var forceModelUnavailable: Bool
    public var forceCapacityExhausted: Bool
    public var forceTimeout: Bool
    public var forceInternalFailureAfterChunks: Int?

    public init(
        modelID: String = "sim-model",
        maxContextTokens: Int = 8192,
        maxOutputTokens: Int = 512,
        isLoaded: Bool = true,
        maximumConcurrentRequests: Int = 2,
        chunkCount: Int = 4,
        chunkDelayNanoseconds: UInt64 = 1_000_000,
        forceModelUnavailable: Bool = false,
        forceCapacityExhausted: Bool = false,
        forceTimeout: Bool = false,
        forceInternalFailureAfterChunks: Int? = nil
    ) {
        self.modelID = modelID
        self.maxContextTokens = maxContextTokens
        self.maxOutputTokens = maxOutputTokens
        self.isLoaded = isLoaded
        self.maximumConcurrentRequests = maximumConcurrentRequests
        self.chunkCount = chunkCount
        self.chunkDelayNanoseconds = chunkDelayNanoseconds
        self.forceModelUnavailable = forceModelUnavailable
        self.forceCapacityExhausted = forceCapacityExhausted
        self.forceTimeout = forceTimeout
        self.forceInternalFailureAfterChunks = forceInternalFailureAfterChunks
    }
}

/// Deterministic, simulation-only implementation of `MLXInferenceRuntime`.
///
/// Used by package tests and by Saturn-Node contract tests that must not download
/// weights or touch SN01 hardware. Live path: `MeshModelInferenceRuntime`.
public actor SimulatedMLXInferenceRuntime: MLXInferenceRuntime {
    private var config: SimulatedInferenceConfig
    private let telemetry: AdapterTelemetry

    private struct ActiveRequest {
        let continuation: AsyncThrowingStream<InferenceChunk, Error>.Continuation
        let task: Task<Void, Never>
        var generatedCount: Int
        let startedAt: Date
        let modelID: String
    }

    private var active: [InferenceRequestID: ActiveRequest] = [:]
    private var terminalIDs: Set<InferenceRequestID> = []

    public init(
        config: SimulatedInferenceConfig = SimulatedInferenceConfig(),
        telemetry: AdapterTelemetry = AdapterTelemetry()
    ) {
        self.config = config
        self.telemetry = telemetry
    }

    public func updateConfig(_ config: SimulatedInferenceConfig) {
        self.config = config
    }

    public func telemetrySnapshot() async -> [AdapterTelemetryRecord] {
        await telemetry.snapshot()
    }

    public func activeRequestIDs() -> Set<InferenceRequestID> {
        Set(active.keys)
    }

    public func capabilities() async throws -> InferenceCapabilities {
        if config.forceModelUnavailable || !config.isLoaded {
            return InferenceCapabilities(
                models: [
                    InferenceModelCapability(
                        modelID: config.modelID,
                        maxContextTokens: config.maxContextTokens,
                        maxOutputTokens: config.maxOutputTokens,
                        isLoaded: false
                    )
                ],
                maximumConcurrentRequests: config.maximumConcurrentRequests,
                state: .unavailable
            )
        }
        if config.forceCapacityExhausted || active.count >= config.maximumConcurrentRequests {
            return InferenceCapabilities(
                models: [
                    InferenceModelCapability(
                        modelID: config.modelID,
                        maxContextTokens: config.maxContextTokens,
                        maxOutputTokens: config.maxOutputTokens,
                        isLoaded: true
                    )
                ],
                maximumConcurrentRequests: config.maximumConcurrentRequests,
                state: .saturated
            )
        }
        return InferenceCapabilities(
            models: [
                InferenceModelCapability(
                    modelID: config.modelID,
                    maxContextTokens: config.maxContextTokens,
                    maxOutputTokens: config.maxOutputTokens,
                    isLoaded: true
                )
            ],
            maximumConcurrentRequests: config.maximumConcurrentRequests,
            state: .available
        )
    }

    public nonisolated func generate(
        _ request: ValidatedInferenceRequest
    ) -> AsyncThrowingStream<InferenceChunk, Error> {
        let pair = AsyncThrowingStream<InferenceChunk, Error>.makeStream(
            bufferingPolicy: .bufferingNewest(32)
        )
        let continuation = pair.continuation

        continuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task {
                await self.consumerTerminated(requestID: request.requestID)
            }
        }

        Task { [weak self] in
            guard let self else {
                continuation.finish(throwing: MeshInferenceError.runtimeUnavailable)
                return
            }
            await self.start(request: request, continuation: continuation)
        }

        return pair.stream
    }

    public func cancel(requestID: InferenceRequestID) async {
        if terminalIDs.contains(requestID) {
            return
        }
        guard let session = active.removeValue(forKey: requestID) else {
            terminalIDs.insert(requestID)
            return
        }
        session.task.cancel()
        session.continuation.yield(.cancelled(requestID: requestID))
        session.continuation.finish()
        terminalIDs.insert(requestID)
        let duration = Date().timeIntervalSince(session.startedAt)
        await telemetry.record(
            AdapterTelemetryRecord(
                requestID: requestID,
                modelID: session.modelID,
                outcome: "cancelled",
                generatedTokenCount: session.generatedCount,
                duration: duration
            )
        )
    }

    // MARK: - Internal

    private func start(
        request: ValidatedInferenceRequest,
        continuation: AsyncThrowingStream<InferenceChunk, Error>.Continuation
    ) async {
        if terminalIDs.contains(request.requestID) {
            continuation.finish(throwing: MeshInferenceError.cancelled)
            return
        }

        if config.forceCapacityExhausted || active.count >= config.maximumConcurrentRequests {
            continuation.finish(throwing: MeshInferenceError.capacityExhausted)
            terminalIDs.insert(request.requestID)
            await telemetry.record(
                AdapterTelemetryRecord(
                    requestID: request.requestID,
                    modelID: request.modelID,
                    outcome: "capacity_exhausted",
                    generatedTokenCount: 0,
                    duration: 0
                )
            )
            return
        }

        let task = Task { [weak self] in
            guard let self else { return }
            await self.runGeneration(request: request, continuation: continuation)
        }

        active[request.requestID] = ActiveRequest(
            continuation: continuation,
            task: task,
            generatedCount: 0,
            startedAt: Date(),
            modelID: request.modelID
        )
    }

    private func runGeneration(
        request: ValidatedInferenceRequest,
        continuation: AsyncThrowingStream<InferenceChunk, Error>.Continuation
    ) async {
        let startedAt = Date()

        if config.forceModelUnavailable || !config.isLoaded {
            continuation.finish(throwing: MeshInferenceError.modelUnavailable(request.modelID))
            await markTerminal(requestID: request.requestID, modelID: request.modelID, outcome: "model_unavailable", count: 0, startedAt: startedAt)
            return
        }
        if request.modelID != config.modelID {
            continuation.finish(throwing: MeshInferenceError.modelUnavailable(request.modelID))
            await markTerminal(requestID: request.requestID, modelID: request.modelID, outcome: "model_unavailable", count: 0, startedAt: startedAt)
            return
        }
        if config.forceTimeout {
            continuation.finish(throwing: MeshInferenceError.requestTimeout)
            await markTerminal(requestID: request.requestID, modelID: request.modelID, outcome: "timeout", count: 0, startedAt: startedAt)
            return
        }

        continuation.yield(.started(requestID: request.requestID))

        let chunks = max(1, min(config.chunkCount, request.maxOutputTokens))
        var generated = 0

        for i in 0..<chunks {
            if Task.isCancelled || terminalIDs.contains(request.requestID) {
                return
            }
            if let failAfter = config.forceInternalFailureAfterChunks, i >= failAfter {
                continuation.finish(throwing: MeshInferenceError.generationFailed("injected internal failure"))
                await markTerminal(requestID: request.requestID, modelID: request.modelID, outcome: "failed", count: generated, startedAt: startedAt)
                return
            }

            if config.chunkDelayNanoseconds > 0 {
                try? await Task.sleep(nanoseconds: config.chunkDelayNanoseconds)
            }

            if Task.isCancelled || terminalIDs.contains(request.requestID) {
                return
            }

            continuation.yield(
                .delta(requestID: request.requestID, text: "tok\(i)", tokenID: i)
            )
            generated += 1
            if var session = active[request.requestID] {
                session.generatedCount = generated
                active[request.requestID] = session
            }
        }

        if Task.isCancelled || terminalIDs.contains(request.requestID) {
            return
        }

        let reason: InferenceFinishReason =
            generated >= request.maxOutputTokens ? .length : .stop
        continuation.yield(.completed(requestID: request.requestID, finishReason: reason))
        continuation.finish()
        await markTerminal(
            requestID: request.requestID,
            modelID: request.modelID,
            outcome: reason == .length ? "length" : "completed",
            count: generated,
            startedAt: startedAt
        )
    }

    private func markTerminal(
        requestID: InferenceRequestID,
        modelID: String,
        outcome: String,
        count: Int,
        startedAt: Date
    ) async {
        active.removeValue(forKey: requestID)
        terminalIDs.insert(requestID)
        let duration = Date().timeIntervalSince(startedAt)
        await telemetry.record(
            AdapterTelemetryRecord(
                requestID: requestID,
                modelID: modelID,
                outcome: outcome,
                generatedTokenCount: count,
                duration: duration
            )
        )
    }

    private func consumerTerminated(requestID: InferenceRequestID) async {
        guard let session = active.removeValue(forKey: requestID) else { return }
        session.task.cancel()
        terminalIDs.insert(requestID)
        let duration = Date().timeIntervalSince(session.startedAt)
        await telemetry.record(
            AdapterTelemetryRecord(
                requestID: requestID,
                modelID: session.modelID,
                outcome: "consumer_cancelled",
                generatedTokenCount: session.generatedCount,
                duration: duration
            )
        )
    }
}
