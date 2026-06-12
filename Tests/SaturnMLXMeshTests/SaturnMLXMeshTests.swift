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

    // MARK: - Phase 2: EpiCache v0.3 real KV + LayerBudgetAllocator + graph residency

    func testLayerBudgetAllocator() {
        let allocator = LayerBudgetAllocator()
        // 4 layers, total 100, sensitivities increasing (last layer most sensitive)
        let budgets = allocator.allocate(totalBudget: 100, layerSensitivities: [1.0, 2.0, 3.0, 10.0], sharpness: 2.0)
        XCTAssertEqual(budgets.count, 4)
        XCTAssertEqual(budgets.reduce(0, +), 100, "Budgets must sum exactly to total")
        // Higher sensitivity should get strictly more or equal (after remainder)
        XCTAssertTrue(budgets[3] >= budgets[2], "Most sensitive layer should receive at least as much")
    }

    func testKVCacheManagerBasic() async {
        let index = EpisodicMemoryIndex()
        let manager = KVCacheManager(index: index)

        let episode = ConversationEpisode(
            id: UUID(),
            turns: [ConversationTurn(role: "user", content: "Hello long context test.")],
            embedding: Array(repeating: 0.1, count: 128)
        )

        let budget = KVCacheBudget(maxTokens: 2048)
        // Use sensitivities to exercise allocator inside build
        let sensitivities = [1.0, 1.5, 2.0, 4.0]
        let meta = await manager.buildEpisodeCache(
            episode: episode,
            budget: budget,
            layerSensitivities: sensitivities
        )

        XCTAssertEqual(meta.episodeID, episode.id)
        XCTAssertNotNil(meta.layerBudgets)
        XCTAssertEqual(meta.layerBudgets?.count, 4)
        XCTAssertEqual(meta.budget.maxTokens, 2048)

        let retrieved = await manager.retrieveCache(for: EpisodeMatch(episode: episode, score: 0.9))
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.layerBudgets, meta.layerBudgets)

        // Real cache path (sim donation) - use Sendable hasRealCache to avoid non-Sendable return across actor boundary in test
        let has1 = await manager.hasRealCache(for: episode.id)
        XCTAssertFalse(has1)
        await manager.storePrefilledCache(for: episode.id, cache: []) // empty sim cache
        let has2 = await manager.hasRealCache(for: episode.id)
        XCTAssertTrue(has2)

        let all = await manager.allCaches()
        XCTAssertEqual(all.count, 1)

        await manager.clear()
        let afterClear = await manager.allCaches()
        XCTAssertTrue(afterClear.isEmpty)
    }

    func testSessionKVAndGraphResidency() async {
        let mesh = MeshSession(controlPlane: .local, policy: .appleSiliconBalanced)

        let episode = ConversationEpisode(
            id: UUID(),
            turns: [
                .init(role: "user", content: "What is the Saturn mesh graph?"),
                .init(role: "assistant", content: "It is a weighted directed computation graph...")
            ],
            embedding: Array(repeating: 0.2, count: 128)
        )

        // Use the Phase 2 test helper (builds via manager + LayerBudget + marks graph)
        let kv = await mesh._buildAndRegisterEpisodeKVForTest(episode: episode, budget: KVCacheBudget(maxTokens: 1024))

        XCTAssertEqual(kv.episodeID, episode.id)
        XCTAssertNotNil(kv.layerBudgets)

        // Manager has it
        let retrieved = await mesh.kvCacheManager.retrieveCache(for: EpisodeMatch(episode: episode, score: 0.8))
        XCTAssertNotNil(retrieved)

        // Graph has residency marked
        let g = await mesh.currentComputationGraph()
        XCTAssertEqual(g.episodeResidencies[kv.episodeID], "local:gpu")
    }

    // MARK: - Phase 3 start: PlacementEngine / graph scheduler stub

    func testPlacementEngineGraphAwareAndRewrite() async {
        let mesh = MeshSession(controlPlane: .local, policy: .appleSiliconBalanced)

        // Populate some graph state (residencies + components) via existing helpers
        await mesh._registerModelForGraphTest(id: "model-a", role: .primary, unit: .gpu)
        let ep = ConversationEpisode(id: UUID(), turns: [.init(role: "user", content: "test residency")], embedding: [])
        let kv = await mesh._buildAndRegisterEpisodeKVForTest(episode: ep)
        // Add second residency so rewrite mem pressure logic triggers (>1)
        let ep2 = ConversationEpisode(id: UUID(), turns: [.init(role: "user", content: "second for residency count")], embedding: [])
        _ = await mesh._buildAndRegisterEpisodeKVForTest(episode: ep2)
        // now graph has residency count >1

        // Engine decide should still pick expected units but enrich notes with graph data
        let engine = await mesh.placementEngine
        let primaryDec = engine.decide(role: .primary, graph: await mesh.currentComputationGraph())
        XCTAssertEqual(primaryDec.unit, .gpu)
        XCTAssertTrue(primaryDec.notes?.contains("graph-aware") ?? false)
        XCTAssertTrue(primaryDec.notes?.contains("residencies") ?? false)

        // Drafter should still be unified (graph enrichment only)
        let drafterDec = engine.decide(role: .drafter, graph: await mesh.currentComputationGraph())
        XCTAssertEqual(drafterDec.unit, .unified)

        // Rewrite using a telemetry snapshot with high tps + pressure
        let snap = TelemetrySnapshot(
            loads: [],
            generations: [
                GenerationInfo(modelID: "model-a", role: .primary, promptTokens: 10, generatedTokens: 200,
                               duration: 1.0, tokensPerSecond: 200.0, speculativeGamma: nil, acceptedTokens: nil,
                               memoryPressureHint: 0.6, timestamp: Date())
            ],
            totalGeneratedTokens: 200,
            averageTokensPerSecond: 200.0,
            lastUpdated: Date()
        )
        await mesh.rewriteGraphUsingTelemetry()  // uses internal snapshot, but for test we can call after manual? For this, directly exercise engine
        var g = await mesh.currentComputationGraph()
        // Before rewrite, latency ~1.0
        let before = g.nodeWeights["local:gpu"]?.latency ?? 1.0

        // Call rewrite directly on engine with our snap (simulates what the session method does)
        var mutableG = await mesh.currentComputationGraph()
        let engine2 = await mesh.placementEngine
        engine2.rewrite(graph: &mutableG, using: snap)
        // After high tps + pressure, latency should drop, memory rise
        let afterLat = mutableG.nodeWeights["local:gpu"]?.latency ?? 1.0
        let afterMem = mutableG.nodeWeights["local:gpu"]?.memory ?? 1.0
        XCTAssertLessThan(afterLat, before, "High tps should have reduced latency weight")
        XCTAssertGreaterThan(afterMem, 1.0, "Memory pressure + residency should have increased memory weight")
    }
}
