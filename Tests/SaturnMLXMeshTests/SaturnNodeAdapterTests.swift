import XCTest
@testable import SaturnMLXMesh

final class SaturnNodeAdapterTests: XCTestCase {

    private func makeRequest(
        id: String = UUID().uuidString,
        model: String = "sim-model",
        maxTokens: Int = 8
    ) -> ValidatedInferenceRequest {
        ValidatedInferenceRequest(
            requestID: InferenceRequestID(id),
            modelID: model,
            prompt: "fixture prompt",
            maxOutputTokens: maxTokens
        )
    }

    private func drain(
        _ stream: AsyncThrowingStream<InferenceChunk, Error>
    ) async throws -> [InferenceChunk] {
        var chunks: [InferenceChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }
        return chunks
    }

    func testPrimaryAcceptancePinIsQwen3_8B_4bit() {
        XCTAssertEqual(AcceptanceModelPin.primaryModelID, "mlx-community/Qwen3-8B-4bit")
        XCTAssertEqual(AcceptanceModelPin.primaryRole, "kf-primary-acceptance")
        XCTAssertGreaterThan(AcceptanceModelPin.smokeMaxTokens, 0)
        XCTAssertFalse(AcceptanceModelPin.primaryModelID.contains("32B"))
    }

    func testNodeContractSurfaceStaysCapabilitiesGenerateCancel() {
        let runtime: any MLXInferenceRuntime = SimulatedMLXInferenceRuntime()
        _ = runtime
        XCTAssertTrue(InferenceChunk.started(requestID: InferenceRequestID("x")).isTerminal == false)
        XCTAssertTrue(InferenceChunk.cancelled(requestID: InferenceRequestID("x")).isTerminal)
    }

    func testNormalMultiChunkCompletion() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(chunkCount: 4, chunkDelayNanoseconds: 0)
        )
        let request = makeRequest()
        let chunks = try await drain(runtime.generate(request))

        XCTAssertEqual(chunks.first, .started(requestID: request.requestID))
        let deltas = chunks.compactMap { chunk -> String? in
            if case let .delta(_, text, _) = chunk { return text }
            return nil
        }
        XCTAssertEqual(deltas, ["tok0", "tok1", "tok2", "tok3"])

        guard case let .completed(id, reason) = chunks.last else {
            return XCTFail("Expected completed terminal")
        }
        XCTAssertEqual(id, request.requestID)
        XCTAssertEqual(reason, .stop)
        XCTAssertEqual(chunks.filter(\.isTerminal).count, 1)
    }

    func testExactlyOneTerminalOnCompletion() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(chunkCount: 2, chunkDelayNanoseconds: 0)
        )
        let request = makeRequest()
        let chunks = try await drain(runtime.generate(request))
        XCTAssertEqual(chunks.filter(\.isTerminal).count, 1)
    }

    func testIdempotentCancellation() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(chunkCount: 20, chunkDelayNanoseconds: 5_000_000)
        )
        let request = makeRequest()
        let stream = runtime.generate(request)

        try await Task.sleep(nanoseconds: 3_000_000)

        await runtime.cancel(requestID: request.requestID)
        await runtime.cancel(requestID: request.requestID)

        var sawCancelled = false
        var terminalCount = 0
        do {
            for try await chunk in stream {
                if chunk.isTerminal {
                    terminalCount += 1
                    if case .cancelled = chunk {
                        sawCancelled = true
                    }
                }
            }
        } catch {}

        XCTAssertTrue(sawCancelled)
        XCTAssertEqual(terminalCount, 1)
        let active = await runtime.activeRequestIDs()
        XCTAssertTrue(active.isEmpty)
    }

    func testTimeoutInjection() async {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(forceTimeout: true)
        )
        let request = makeRequest()
        do {
            _ = try await drain(runtime.generate(request))
            XCTFail("Expected timeout")
        } catch let error as MeshInferenceError {
            XCTAssertEqual(error, .requestTimeout)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
        let active = await runtime.activeRequestIDs()
        XCTAssertTrue(active.isEmpty)
    }

    func testModelUnavailable() async {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(forceModelUnavailable: true)
        )
        let request = makeRequest()
        do {
            _ = try await drain(runtime.generate(request))
            XCTFail("Expected modelUnavailable")
        } catch let error as MeshInferenceError {
            if case .modelUnavailable = error {
                // ok
            } else {
                XCTFail("Wrong error: \(error)")
            }
        } catch {
            XCTFail("Unexpected: \(error)")
        }
    }

    func testUnknownModelID() async {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(modelID: "sim-model")
        )
        let request = makeRequest(model: "other-model")
        do {
            _ = try await drain(runtime.generate(request))
            XCTFail("Expected modelUnavailable")
        } catch let error as MeshInferenceError {
            if case .modelUnavailable = error {
                // ok
            } else {
                XCTFail("Wrong error: \(error)")
            }
        } catch {
            XCTFail("Unexpected: \(error)")
        }
    }

    func testCapacityExhausted() async {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(forceCapacityExhausted: true)
        )
        let request = makeRequest()
        do {
            _ = try await drain(runtime.generate(request))
            XCTFail("Expected capacityExhausted")
        } catch let error as MeshInferenceError {
            XCTAssertEqual(error, .capacityExhausted)
        } catch {
            XCTFail("Unexpected: \(error)")
        }
        let active = await runtime.activeRequestIDs()
        XCTAssertTrue(active.isEmpty)
    }

    func testCleanupAfterStreamConsumerCancellation() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(chunkCount: 50, chunkDelayNanoseconds: 5_000_000)
        )
        let request = makeRequest()
        let stream = runtime.generate(request)

        var iterator = stream.makeAsyncIterator()
        _ = try await iterator.next()

        await runtime.cancel(requestID: request.requestID)

        try await Task.sleep(nanoseconds: 5_000_000)
        let active = await runtime.activeRequestIDs()
        XCTAssertTrue(active.isEmpty, "No orphan generation after consumer cancel")
    }

    func testCleanupAfterInternalFailure() async {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(
                chunkCount: 10,
                chunkDelayNanoseconds: 0,
                forceInternalFailureAfterChunks: 2
            )
        )
        let request = makeRequest()
        do {
            _ = try await drain(runtime.generate(request))
            XCTFail("Expected generationFailed")
        } catch let error as MeshInferenceError {
            if case .generationFailed = error {
                // ok
            } else {
                XCTFail("Wrong error: \(error)")
            }
        } catch {
            XCTFail("Unexpected: \(error)")
        }
        let active = await runtime.activeRequestIDs()
        XCTAssertTrue(active.isEmpty)
    }

    func testSubsequentRequestSucceedsAfterCancellation() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(chunkCount: 20, chunkDelayNanoseconds: 3_000_000)
        )
        let first = makeRequest(id: "cancel-then-ok-1")
        let stream = runtime.generate(first)
        try await Task.sleep(nanoseconds: 2_000_000)
        await runtime.cancel(requestID: first.requestID)
        _ = try? await drain(stream)

        await runtime.updateConfig(
            SimulatedInferenceConfig(chunkCount: 3, chunkDelayNanoseconds: 0)
        )
        let second = makeRequest(id: "cancel-then-ok-2")
        let chunks = try await drain(runtime.generate(second))
        XCTAssertTrue(chunks.contains {
            if case .completed = $0 { return true }
            return false
        })
        XCTAssertEqual(chunks.filter(\.isTerminal).count, 1)
    }

    func testSubsequentRequestSucceedsAfterFailure() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(
                chunkCount: 5,
                chunkDelayNanoseconds: 0,
                forceInternalFailureAfterChunks: 1
            )
        )
        let first = makeRequest(id: "fail-then-ok-1")
        do {
            _ = try await drain(runtime.generate(first))
            XCTFail("Expected failure")
        } catch {
            // expected
        }

        await runtime.updateConfig(
            SimulatedInferenceConfig(chunkCount: 2, chunkDelayNanoseconds: 0)
        )
        let second = makeRequest(id: "fail-then-ok-2")
        let chunks = try await drain(runtime.generate(second))
        XCTAssertTrue(chunks.contains {
            if case .completed = $0 { return true }
            return false
        })
    }

    func testTelemetryExcludesPromptAndGeneratedText() async throws {
        let telemetry = AdapterTelemetry()
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(chunkCount: 3, chunkDelayNanoseconds: 0),
            telemetry: telemetry
        )
        let request = makeRequest()
        _ = try await drain(runtime.generate(request))

        let records = await runtime.telemetrySnapshot()
        XCTAssertEqual(records.count, 1)
        let rec = records[0]
        XCTAssertEqual(rec.requestID, request.requestID)
        XCTAssertEqual(rec.modelID, "sim-model")
        XCTAssertEqual(rec.outcome, "completed")
        XCTAssertEqual(rec.generatedTokenCount, 3)

        let mirror = Mirror(reflecting: rec)
        let labels = mirror.children.compactMap { $0.label }
        XCTAssertFalse(labels.contains("prompt"))
        XCTAssertFalse(labels.contains("text"))
        XCTAssertFalse(labels.contains("generatedText"))
        XCTAssertTrue(labels.contains("outcome"))
        XCTAssertTrue(labels.contains("generatedTokenCount"))
    }

    func testCapabilitiesReportsLoadedModel() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(
                modelID: "sim-model",
                maxContextTokens: 4096,
                maxOutputTokens: 256,
                isLoaded: true,
                maximumConcurrentRequests: 3
            )
        )
        let caps = try await runtime.capabilities()
        XCTAssertEqual(caps.state, .available)
        XCTAssertEqual(caps.maximumConcurrentRequests, 3)
        XCTAssertEqual(caps.models.count, 1)
        XCTAssertEqual(caps.models[0].modelID, "sim-model")
        XCTAssertTrue(caps.models[0].isLoaded)
    }

    func testNoOrphanAfterNormalCompletion() async throws {
        let runtime = SimulatedMLXInferenceRuntime(
            config: SimulatedInferenceConfig(chunkCount: 2, chunkDelayNanoseconds: 0)
        )
        let request = makeRequest()
        _ = try await drain(runtime.generate(request))
        let active = await runtime.activeRequestIDs()
        XCTAssertTrue(active.isEmpty)
    }
}
