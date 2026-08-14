// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// MeshModelInferenceRuntime.swift
// Real MLX-backed implementation of MLXInferenceRuntime.
//
// This is the non-simulation path: load weights via MeshSession, stream tokens
// from MeshModel.generate, map into InferenceChunk for Saturn-Node.
//
// CI and unit tests must continue to use SimulatedMLXInferenceRuntime.
// Constructing this type downloads/loads model weights and requires Apple Silicon.

import Foundation

/// Real-hardware / real-weights implementation of `MLXInferenceRuntime`.
///
/// - Loads one `MeshModel` through `MeshSession.loadModel`
/// - Streams generation via `MeshModel.generate`
/// - Supports cooperative cancellation of the active request task
///
/// Not selected by Saturn-Node default composition. Call `loadPrimary()` (or
/// the designated initializer after an external load) only under explicit opt-in.
public actor MeshModelInferenceRuntime: MLXInferenceRuntime {
    private let modelID: String
    private let model: MeshModel
    private let maxContextTokens: Int
    private let maxOutputTokensCap: Int
    private let maximumConcurrentRequests: Int
    private let telemetry: AdapterTelemetry

    private struct ActiveRequest {
        let continuation: AsyncThrowingStream<InferenceChunk, Error>.Continuation
        let task: Task<Void, Never>
        var generatedCount: Int
        let startedAt: Date
    }

    private var active: [InferenceRequestID: ActiveRequest] = [:]
    private var terminalIDs: Set<InferenceRequestID> = []

    /// Load the KF / mesh#1 primary model and return a ready runtime.
    ///
    /// Downloads weights on first use if not cached. Requires Apple Silicon + network
    /// (or local HF cache). Do not call from CI unit tests.
    public static func loadPrimary(
        modelID: String = AcceptanceModelPin.primaryModelID,
        maxContextTokens: Int = 8192,
        maxOutputTokensCap: Int = 1024,
        maximumConcurrentRequests: Int = 1,
        telemetry: AdapterTelemetry = AdapterTelemetry()
    ) async throws -> MeshModelInferenceRuntime {
        let session = MeshSession(
            controlPlane: .local,
            policy: .appleSiliconBalanced
        )
        let model = try await session.loadModel(id: modelID, role: .primary)
        return MeshModelInferenceRuntime(
            modelID: modelID,
            model: model,
            maxContextTokens: maxContextTokens,
            maxOutputTokensCap: maxOutputTokensCap,
            maximumConcurrentRequests: maximumConcurrentRequests,
            telemetry: telemetry
        )
    }

    public init(
        modelID: String,
        model: MeshModel,
        maxContextTokens: Int = 8192,
        maxOutputTokensCap: Int = 1024,
        maximumConcurrentRequests: Int = 1,
        telemetry: AdapterTelemetry = AdapterTelemetry()
    ) {
        self.modelID = modelID
        self.model = model
        self.maxContextTokens = maxContextTokens
        self.maxOutputTokensCap = maxOutputTokensCap
        self.maximumConcurrentRequests = max(1, maximumConcurrentRequests)
        self.telemetry = telemetry
    }

    public func telemetrySnapshot() async -> [AdapterTelemetryRecord] {
        await telemetry.snapshot()
    }

    public func activeRequestIDs() -> Set<InferenceRequestID> {
        Set(active.keys)
    }

    // MARK: - MLXInferenceRuntime

    public func capabilities() async throws -> InferenceCapabilities {
        let saturated = active.count >= maximumConcurrentRequests
        return InferenceCapabilities(
            models: [
                InferenceModelCapability(
                    modelID: modelID,
                    maxContextTokens: maxContextTokens,
                    maxOutputTokens: maxOutputTokensCap,
                    isLoaded: true
                )
            ],
            maximumConcurrentRequests: maximumConcurrentRequests,
            state: saturated ? .saturated : .available
        )
    }

    public nonisolated func generate(
        _ request: ValidatedInferenceRequest
    ) -> AsyncThrowingStream<InferenceChunk, Error> {
        let pair = AsyncThrowingStream<InferenceChunk, Error>.makeStream(
            bufferingPolicy: .bufferingNewest(64)
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
        terminalIDs.insert(requestID)
        session.continuation.yield(.cancelled(requestID: requestID))
        session.continuation.finish()
        let duration = Date().timeIntervalSince(session.startedAt)
        await telemetry.record(
            AdapterTelemetryRecord(
                requestID: requestID,
                modelID: modelID,
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

        if request.modelID != modelID {
            terminalIDs.insert(request.requestID)
            await telemetry.record(
                AdapterTelemetryRecord(
                    requestID: request.requestID,
                    modelID: request.modelID,
                    outcome: "model_unavailable",
                    generatedTokenCount: 0,
                    duration: 0
                )
            )
            continuation.finish(throwing: MeshInferenceError.modelUnavailable(request.modelID))
            return
        }

        if active.count >= maximumConcurrentRequests {
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
            continuation.finish(throwing: MeshInferenceError.capacityExhausted)
            return
        }

        let maxTokens = min(max(1, request.maxOutputTokens), maxOutputTokensCap)
        let temperature = request.temperature ?? 0.7
        let startedAt = Date()

        let task = Task { [weak self] in
            guard let self else { return }
            await self.runGeneration(
                request: request,
                maxTokens: maxTokens,
                temperature: temperature,
                continuation: continuation,
                startedAt: startedAt
            )
        }

        active[request.requestID] = ActiveRequest(
            continuation: continuation,
            task: task,
            generatedCount: 0,
            startedAt: startedAt
        )
    }

    private func runGeneration(
        request: ValidatedInferenceRequest,
        maxTokens: Int,
        temperature: Float,
        continuation: AsyncThrowingStream<InferenceChunk, Error>.Continuation,
        startedAt: Date
    ) async {
        continuation.yield(.started(requestID: request.requestID))

        do {
            let tokenStream = try await model.generate(
                prompt: request.prompt,
                maxTokens: maxTokens,
                temperature: temperature
            )

            var generated = 0
            for try await token in tokenStream {
                if Task.isCancelled || terminalIDs.contains(request.requestID) {
                    return
                }
                guard !token.text.isEmpty else { continue }
                continuation.yield(
                    .delta(
                        requestID: request.requestID,
                        text: token.text,
                        tokenID: token.tokenID
                    )
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
                generated >= maxTokens ? .length : .stop

            // Remove request-owned state before finishing the continuation. The
            // stream's onTermination callback also runs for normal finish; if we
            // finished first, it could race and misclassify a completed request as
            // consumer-cancelled or emit duplicate telemetry.
            await markTerminal(
                requestID: request.requestID,
                outcome: reason == .length ? "length" : "completed",
                count: generated,
                startedAt: startedAt
            )
            continuation.yield(
                .completed(requestID: request.requestID, finishReason: reason)
            )
            continuation.finish()
        } catch is CancellationError {
            // cancel(requestID:) owns the terminal chunk when it raced us.
            if !terminalIDs.contains(request.requestID) {
                let generated = active[request.requestID]?.generatedCount ?? 0
                await markTerminal(
                    requestID: request.requestID,
                    outcome: "cancelled",
                    count: generated,
                    startedAt: startedAt
                )
                continuation.yield(.cancelled(requestID: request.requestID))
                continuation.finish()
            }
        } catch {
            let generated = active[request.requestID]?.generatedCount ?? 0
            await markTerminal(
                requestID: request.requestID,
                outcome: "failed",
                count: generated,
                startedAt: startedAt
            )
            continuation.finish(
                throwing: MeshInferenceError.generationFailed(String(describing: error))
            )
        }
    }

    /// Mark the request terminal before any continuation finish so normal stream
    /// termination cannot race with `consumerTerminated` and overwrite its outcome.
    private func markTerminal(
        requestID: InferenceRequestID,
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
                modelID: modelID,
                outcome: "consumer_cancelled",
                generatedTokenCount: session.generatedCount,
                duration: duration
            )
        )
    }
}
