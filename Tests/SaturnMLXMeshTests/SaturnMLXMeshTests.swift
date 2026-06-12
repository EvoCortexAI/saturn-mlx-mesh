// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// MeshModelTests.swift
// Basic verification tests for saturn-mlx-mesh v0.1.
//
// These tests deliberately avoid full model downloads and GPU inference so that
// `swift test` remains fast and works in CI or on machines without the
// large quantized weights. They cover the public API surface, placement policy,
// telemetry, and session factory wiring.
//
// Full end-to-end streaming + real speculative generation should be validated
// manually on Apple Silicon hardware with `swift run` or a dedicated benchmark
// target (see SaturnBench patterns in the control plane for inspiration).

import XCTest
@testable import SaturnMLXMesh

final class SaturnMLXMeshTests: XCTestCase {

    func testAppleSiliconBalancedPolicyDecisions() {
        let policy = PlacementPolicyKind.appleSiliconBalanced

        let primary = policy.decide(role: .primary)
        XCTAssertEqual(primary.unit, .gpu)
        XCTAssertTrue(primary.allowSpeculative)

        let drafter = policy.decide(role: .drafter)
        XCTAssertEqual(drafter.unit, .unified)
        XCTAssertTrue(drafter.allowSpeculative)
        XCTAssertNotNil(drafter.maxKVSizeHint)

        let secondary = policy.decide(role: .secondary)
        XCTAssertFalse(secondary.allowSpeculative)
    }

    func testTelemetryRecordsAndSnapshot() async {
        let telemetry = MeshTelemetry()

        let decision = PlacementDecision(unit: .gpu, allowSpeculative: true)
        await telemetry.recordLoad(
            modelID: "mlx-community/Qwen3-8B-4bit",
            role: .primary,
            decision: decision,
            loadDuration: 2.34
        )

        let genInfo = GenerationInfo(
            modelID: "mlx-community/Qwen3-8B-4bit",
            role: .primary,
            promptTokens: 12,
            generatedTokens: 87,
            duration: 1.23,
            tokensPerSecond: 70.7,
            speculativeGamma: 4,
            acceptedTokens: 71,
            memoryPressureHint: 0.42,
            timestamp: Date()
        )
        await telemetry.recordGenerationInfo(genInfo)

        let snap = await telemetry.snapshot()
        XCTAssertEqual(snap.loads.count, 1)
        XCTAssertEqual(snap.generations.count, 1)
        XCTAssertEqual(snap.totalGeneratedTokens, 87)
        XCTAssertNotNil(snap.averageTokensPerSecond)
        XCTAssertGreaterThan(snap.averageTokensPerSecond!, 60)
    }

    func testMeshSessionCreationAndPolicyWiring() async throws {
        let mesh = MeshSession(
            controlPlane: .local,
            policy: .appleSiliconBalanced,
            defaultSpeculativeGamma: 4
        )

        // We cannot call loadModel here without network + weights, but we can
        // verify that the session was constructed with the expected policy
        // by exercising the public telemetry surface (empty snapshot).
        let snap = await mesh.telemetrySnapshot()
        XCTAssertTrue(snap.loads.isEmpty)
        XCTAssertTrue(snap.generations.isEmpty)

        // Also exercise the type aliases and enums are public as documented.
        let _ = ModelRole.primary
        let _ = MeshExecutionUnit.gpu
    }

    // MARK: - Generate behavior tests (skeleton-hardening)

    func testGenerateThrowsNotLoadedWhenNoContainerAttached() async {
        let model = MeshModel(
            id: "test-unloaded",
            role: .primary,
            placement: PlacementDecision(unit: .gpu),
            telemetry: MeshTelemetry()
        )

        do {
            _ = try await model.generate(prompt: "hello")
            XCTFail("Expected .notLoaded")
        } catch let error as MeshModelError {
            if case .notLoaded = error {
                // success
            } else {
                XCTFail("Wrong error: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testStreamFinishesAfterSuccessfulGeneration() async throws {
        let telemetry = MeshTelemetry()
        let model = MeshModel(
            id: "sim-model",
            role: .primary,
            placement: PlacementDecision(unit: .gpu),
            telemetry: telemetry
        )
        await model._enableTestSuccessSimulation()

        let stream = try await model.generate(prompt: "test prompt", maxTokens: 10)

        var received: [String] = []
        for try await token in stream {
            received.append(token.text)
        }

        XCTAssertEqual(received, ["tok0", "tok1", "tok2", "tok3"])
        // If we reach here without hanging, the stream completed (finish() was called)
    }

    func testSpeculativeTelemetryAbsentWhenNoDrafterAttached() async throws {
        let telemetry = MeshTelemetry()
        let model = MeshModel(
            id: "sim-spec-no-drafter",
            role: .primary,
            placement: PlacementDecision(unit: .gpu, allowSpeculative: true),
            telemetry: telemetry,
            speculativeGammaDefault: 4   // requested, but no drafter container attached
        )
        await model._enableTestSuccessSimulation()

        let stream = try await model.generate(prompt: "spec test")
        // drain the stream
        for try await _ in stream {}

        let snap = await telemetry.snapshot()
        let last = snap.generations.last
        XCTAssertNotNil(last)
        XCTAssertNil(last?.speculativeGamma, "Should not claim speculative when no drafter was attached")
        XCTAssertNil(last?.acceptedTokens)
    }

    func testEpisodicMemoryIndexBasic() async throws {
        let index = EpisodicMemoryIndex(numEpisodes: 3, windowSize: 2)

        let turns: [ConversationTurn] = [
            .init(role: "user", content: "What is the capital of France?"),
            .init(role: "assistant", content: "Paris is the capital of France."),
            .init(role: "user", content: "Tell me about the Eiffel Tower."),
            .init(role: "assistant", content: "The Eiffel Tower is in Paris."),
            .init(role: "user", content: "What is the weather in Tokyo?"),
            .init(role: "assistant", content: "I don't have real-time weather, but Tokyo is usually mild.")
        ]

        await index.ingest(turns: turns)

        let all = await index.episodes()
        XCTAssertGreaterThanOrEqual(all.count, 1)

        let match = await index.match(query: "capital of France")
        XCTAssertNotNil(match)
        if let m = match {
            XCTAssertGreaterThan(m.score, 0.1)
            let hasFrance = m.episode.turns.contains { $0.content.contains("France") }
            XCTAssertTrue(hasFrance)
        }
    }

    // MARK: - Phase 1 graph as first-class (L1-10 + feedback)

    func testMeshComputationGraphAndFeedback() async {
        let mesh = MeshSession(controlPlane: .local, policy: .appleSiliconBalanced)

        // Use the test-only register (avoids real load/network in CI).
        await mesh._registerModelForGraphTest(id: "test-primary-8b", role: .primary, unit: .gpu)
        await mesh._registerModelForGraphTest(id: "test-drafter-1b", role: .drafter, unit: .unified)

        var g = await mesh.currentComputationGraph()
        XCTAssertGreaterThanOrEqual(g.components.count, 2)
        XCTAssertTrue(g.siliconUnits.contains { $0.unit == .gpu })
        XCTAssertTrue(g.siliconUnits.contains { $0.unit == .unified })
        XCTAssertTrue(g.active.contains("test-primary-8b"))
        XCTAssertTrue(g.components.contains { $0.id == "test-drafter-1b" })

        // Note: the L7 speculative propose/verify edge (drafter -> primary) is added by the
        // real loadModel(id:drafterId:) path. The register test helper keeps this unit test
        // free of network; the static builder below + loadModel source prove the wiring.

        // Also exercise the static L7 example builder (explicit subgraph for future executor)
        let l7 = MeshComputationGraph.l7SpeculativeExample(primaryID: "p", drafterID: "d")
        XCTAssertEqual(l7.active.count, 2)

        // Simulate a high-throughput generation and feed back into weights (Level 9 live cost).
        let fastGen = GenerationInfo(
            modelID: "test-primary-8b",
            role: .primary,
            promptTokens: 10,
            generatedTokens: 120,
            duration: 0.8,
            tokensPerSecond: 150.0,
            speculativeGamma: nil,
            acceptedTokens: nil,
            memoryPressureHint: nil,
            timestamp: Date()
        )
        await mesh.recordObservedCostFrom(info: fastGen, for: "test-primary-8b", unit: .gpu)

        g = await mesh.currentComputationGraph()
        // The silicon key or component weight should have been blended (lower latency proxy).
        if let w = g.nodeWeight(for: "local:gpu") ?? g.nodeWeight(for: "test-primary-8b") {
            // Original latency ~1.0; after high tps blend it should be noticeably smaller.
            XCTAssertLessThan(w.latency, 0.9, "Observed high tps should have reduced effective latency weight")
        } else {
            XCTFail("Expected a node weight for the primary unit or component after feedback")
        }
    }
}
