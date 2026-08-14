// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// SaturnNodeAdapter.swift
// Narrow, stable library-side adapter *contract* for Saturn-Node.
//
// Implementations (separate files):
//   - SimulatedMLXInferenceRuntime  — CI / unit tests (no weights)
//   - MeshModelInferenceRuntime     — real MLX path (opt-in)
//
// Saturn-Node owns workload authentication, policy, quotas, network transport,
// and service lifecycle. This surface receives only already-validated inference
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
///
/// Real path: `MeshModelInferenceRuntime`. CI path: `SimulatedMLXInferenceRuntime`.
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
