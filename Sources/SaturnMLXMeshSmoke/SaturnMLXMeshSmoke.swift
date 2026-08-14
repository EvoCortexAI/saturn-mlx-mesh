// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// SaturnMLXMeshSmoke
//
// Opt-in real-hardware acceptance executable.
//
// Ordinary `swift test` / CI never invokes this target and therefore never
// downloads model weights. Run it only on an explicitly selected Apple Silicon
// acceptance host.
//
// Baseline completion + timing:
//     swift run SaturnMLXMeshSmoke
//
// Add explicit cancellation + subsequent-request recovery:
//     swift run SaturnMLXMeshSmoke --cancel-recovery
//
// Generated content is suppressed by default so copied acceptance output stays
// metadata-only. `--show-content` is a local debugging aid and should not be used
// for standard evidence artifacts.

import Foundation
import SaturnMLXMesh

@main
struct SaturnMLXMeshSmoke {
    private enum SmokeFailure: Error, CustomStringConvertible {
        case modelNotLoaded(String)
        case noGeneratedDeltas
        case missingCompletion
        case unexpectedCancellation
        case cancellationNotObserved
        case requestStillActive

        var description: String {
            switch self {
            case let .modelNotLoaded(modelID):
                return "Pinned model is not reported loaded: \(modelID)"
            case .noGeneratedDeltas:
                return "Real MLX stream produced no non-empty deltas"
            case .missingCompletion:
                return "Real MLX stream ended without a completed terminal event"
            case .unexpectedCancellation:
                return "Baseline completion request was cancelled unexpectedly"
            case .cancellationNotObserved:
                return "Cancellation probe did not observe a cancelled terminal event"
            case .requestStillActive:
                return "Cancelled request remained active after stream termination"
            }
        }
    }

    private struct CompletionMetrics {
        let deltaCount: Int
        let timeToFirstDelta: Duration
        let generationDuration: Duration
        let finishReason: InferenceFinishReason
    }

    static func main() async throws {
        let arguments = Set(CommandLine.arguments.dropFirst())
        let showContent = arguments.contains("--show-content")
        let runCancellationRecovery = arguments.contains("--cancel-recovery")

        let modelID = AcceptanceModelPin.primaryModelID
        let clock = ContinuousClock()

        print("=== saturn-mlx-mesh hardware acceptance ===")
        print("model_id=\(modelID)")
        print("model_role=\(AcceptanceModelPin.primaryRole)")
        print("max_output_tokens=\(AcceptanceModelPin.smokeMaxTokens)")
        print("content_output=\(showContent ? "enabled" : "suppressed")")

        let loadStarted = clock.now
        let runtime = try await MeshModelInferenceRuntime.loadPrimary()
        let loadDuration = loadStarted.duration(to: clock.now)

        let capabilities = try await runtime.capabilities()
        guard capabilities.models.contains(where: {
            $0.modelID == modelID && $0.isLoaded
        }) else {
            throw SmokeFailure.modelNotLoaded(modelID)
        }

        print("load_ms=\(milliseconds(loadDuration))")
        print("runtime_state=\(capabilities.state.rawValue)")
        print("maximum_concurrent_requests=\(capabilities.maximumConcurrentRequests)")

        let baseline = try await runCompletion(
            runtime: runtime,
            requestID: InferenceRequestID("hardware-baseline-\(UUID().uuidString)"),
            showContent: showContent
        )

        print("baseline_result=pass")
        print("baseline_delta_count=\(baseline.deltaCount)")
        print("baseline_ttfd_ms=\(milliseconds(baseline.timeToFirstDelta))")
        print("baseline_generation_ms=\(milliseconds(baseline.generationDuration))")
        print("baseline_finish_reason=\(baseline.finishReason.rawValue)")

        if runCancellationRecovery {
            try await runCancellationProbe(runtime: runtime, showContent: showContent)

            let recovery = try await runCompletion(
                runtime: runtime,
                requestID: InferenceRequestID("hardware-recovery-\(UUID().uuidString)"),
                showContent: showContent
            )

            print("cancel_recovery_result=pass")
            print("recovery_delta_count=\(recovery.deltaCount)")
            print("recovery_ttfd_ms=\(milliseconds(recovery.timeToFirstDelta))")
            print("recovery_generation_ms=\(milliseconds(recovery.generationDuration))")
            print("recovery_finish_reason=\(recovery.finishReason.rawValue)")
        }

        let telemetry = await runtime.telemetrySnapshot()
        print("telemetry_records=\(telemetry.count)")
        print("result=pass")
    }

    private static func runCompletion(
        runtime: MeshModelInferenceRuntime,
        requestID: InferenceRequestID,
        showContent: Bool
    ) async throws -> CompletionMetrics {
        let request = ValidatedInferenceRequest(
            requestID: requestID,
            modelID: AcceptanceModelPin.primaryModelID,
            prompt: AcceptanceModelPin.acceptancePrompt,
            maxOutputTokens: AcceptanceModelPin.smokeMaxTokens,
            temperature: 0.7
        )

        let clock = ContinuousClock()
        let started = clock.now
        var firstDeltaAt: ContinuousClock.Instant?
        var deltaCount = 0
        var finishReason: InferenceFinishReason?

        let stream = runtime.generate(request)
        for try await chunk in stream {
            switch chunk {
            case .started:
                break
            case let .delta(_, text, _):
                guard !text.isEmpty else { continue }
                if firstDeltaAt == nil {
                    firstDeltaAt = clock.now
                }
                deltaCount += 1
                if showContent {
                    print(text, terminator: "")
                    fflush(stdout)
                }
            case let .completed(_, reason):
                finishReason = reason
            case .cancelled:
                throw SmokeFailure.unexpectedCancellation
            }
        }

        if showContent {
            print("")
        }

        guard deltaCount > 0, let firstDeltaAt else {
            throw SmokeFailure.noGeneratedDeltas
        }
        guard let finishReason else {
            throw SmokeFailure.missingCompletion
        }

        return CompletionMetrics(
            deltaCount: deltaCount,
            timeToFirstDelta: started.duration(to: firstDeltaAt),
            generationDuration: started.duration(to: clock.now),
            finishReason: finishReason
        )
    }

    private static func runCancellationProbe(
        runtime: MeshModelInferenceRuntime,
        showContent: Bool
    ) async throws {
        let requestID = InferenceRequestID("hardware-cancel-\(UUID().uuidString)")
        let request = ValidatedInferenceRequest(
            requestID: requestID,
            modelID: AcceptanceModelPin.primaryModelID,
            prompt: AcceptanceModelPin.acceptancePrompt,
            maxOutputTokens: max(AcceptanceModelPin.smokeMaxTokens, 64),
            temperature: 0.7
        )

        var cancellationRequested = false
        var cancellationObserved = false
        let stream = runtime.generate(request)

        for try await chunk in stream {
            switch chunk {
            case .started:
                break
            case let .delta(_, text, _):
                if showContent, !text.isEmpty {
                    print(text, terminator: "")
                    fflush(stdout)
                }
                if !cancellationRequested {
                    cancellationRequested = true
                    await runtime.cancel(requestID: requestID)
                }
            case let .completed(_, reason):
                if reason == .cancelled {
                    cancellationObserved = true
                }
            case .cancelled:
                cancellationObserved = true
            }
        }

        if showContent {
            print("")
        }

        guard cancellationRequested, cancellationObserved else {
            throw SmokeFailure.cancellationNotObserved
        }
        guard await runtime.activeRequestIDs().isEmpty else {
            throw SmokeFailure.requestStillActive
        }

        print("cancel_result=pass")
    }

    private static func milliseconds(_ duration: Duration) -> String {
        let components = duration.components
        let value = Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
        return String(format: "%.3f", value)
    }
}
