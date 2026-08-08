// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// SaturnNodeAdapter.swift
// Narrow, stable library-side adapter surface for Saturn-Node.
//
// Saturn-Node owns workload authentication, policy, quotas, network transport,
// and service lifecycle. This adapter receives only already-validated inference
// requests and never parses credentials or opens listeners.
//
// Graph, placement, speculative-decoding, episodic memory, and Runtime DAG types
// are intentionally excluded from this contract.

import Foundation

// MARK: - Identifiers and request

/// Opaque request correlation identifier owned by the caller (Saturn-Node).
public struct InferenceRequestID: Hashable, Sendable, Codable {
    public let rawValue: String

    public init(_ rawValue: String) {
        self.rawValue = rawValue
    }
}

/// Already-validated inference request. Saturn-Node must enforce model allowlist,
/// context/output bounds, concurrency, and budget before constructing this value.
public struct ValidatedInferenceRequest: Sendable, Hashable {
    public let requestID: InferenceRequestID
    public let modelID: String
    public let prompt: String
    public let maxOutputTokens: Int
    public let temperature: Float?

    public init(
        requestID: InferenceRequestID,
        modelID: String,
        prompt: String,
        maxOutputTokens: Int,
        temperature: Float? = nil
    ) {
        self.requestID = requestID
        self.modelID = modelID
        self.prompt = prompt
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
    }
}

// MARK: - Capabilities

public enum InferenceRuntimeState: String, Sendable, Hashable, Codable {
    case available
    case saturated
    case unavailable
}

public struct InferenceModelCapability: Sendable, Hashable, Codable {
    public let modelID: String
    public let maxContextTokens: Int
    public let maxOutputTokens: Int
    public let isLoaded: Bool

    public init(
        modelID: String,
        maxContextTokens: Int,
        maxOutputTokens: Int,
        isLoaded: Bool
    ) {
        self.modelID = modelID
        self.maxContextTokens = maxContextTokens
        self.maxOutputTokens = maxOutputTokens
        self.isLoaded = isLoaded
    }
}

public struct InferenceCapabilities: Sendable, Hashable, Codable {
    public let models: [InferenceModelCapability]
    public let maximumConcurrentRequests: Int
    public let state: InferenceRuntimeState

    public init(
        models: [InferenceModelCapability],
        maximumConcurrentRequests: Int,
        state: InferenceRuntimeState
    ) {
        self.models = models
        self.maximumConcurrentRequests = maximumConcurrentRequests
        self.state = state
    }
}

// MARK: - Stream chunks and outcomes

public enum InferenceFinishReason: String, Sendable, Hashable, Codable {
    case stop
    case length
    case cancelled
}

/// Typed incremental events emitted by the library adapter.
/// Prompt and generated text appear only in `.delta`; telemetry must not log them.
public enum InferenceChunk: Sendable, Hashable {
    case started(requestID: InferenceRequestID)
    case delta(requestID: InferenceRequestID, text: String, tokenID: Int?)
    case completed(requestID: InferenceRequestID, finishReason: InferenceFinishReason)
    case cancelled(requestID: InferenceRequestID)

    public var requestID: InferenceRequestID {
        switch self {
        case let .started(id), let .delta(id, _, _), let .completed(id, _), let .cancelled(id):
            return id
        }
    }

    public var isTerminal: Bool {
        switch self {
        case .completed, .cancelled: return true
        case .started, .delta: return false
        }
    }
}

// MARK: - Errors

public enum MeshInferenceError: Error, Sendable, Hashable {
    case modelUnavailable(String)
    case capacityExhausted
    case requestTimeout
    case cancelled
    case notLoaded
    case generationFailed(String)
    case runtimeUnavailable
}

// MARK: - Protocol

/// Stable library-side surface for Saturn-Node.
///
/// Implementations must:
/// - emit exactly one terminal chunk (completed or cancelled) or throw;
/// - make cancel(requestID:) idempotent;
/// - release request resources so a subsequent request can succeed;
/// - keep telemetry free of prompt and generated text.
public protocol MLXInferenceRuntime: Sendable {
    func capabilities() async throws -> InferenceCapabilities

    func generate(
        _ request: ValidatedInferenceRequest
    ) -> AsyncThrowingStream<InferenceChunk, Error>

    func cancel(requestID: InferenceRequestID) async
}

// MARK: - Metadata-only telemetry for the adapter

public struct AdapterTelemetryRecord: Sendable, Hashable {
    public let requestID: InferenceRequestID
    public let modelID: String
    public let outcome: String
    public let generatedTokenCount: Int
    public let duration: TimeInterval
    public let timestamp: Date

    public init(
        requestID: InferenceRequestID,
        modelID: String,
        outcome: String,
        generatedTokenCount: Int,
        duration: TimeInterval,
        timestamp: Date = Date()
    ) {
        self.requestID = requestID
        self.modelID = modelID
        self.outcome = outcome
        self.generatedTokenCount = generatedTokenCount
        self.duration = duration
        self.timestamp = timestamp
    }
}

public actor AdapterTelemetry {
    private var records: [AdapterTelemetryRecord] = []

    public init() {}

    public func record(_ entry: AdapterTelemetryRecord) {
        records.append(entry)
    }

    public func snapshot() -> [AdapterTelemetryRecord] {
        records
    }

    public func clear() {
        records.removeAll()
    }
}

// MARK: - Simulated runtime (deterministic, no MLX)

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
/// weights or touch SN01 hardware. Production composition selects a different
/// implementation that drives a loaded `MeshModel`.
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

    public func generate(
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

        // Kick off work on the actor so registration happens before generation proceeds.
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

        // Capacity gate before registration.
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
