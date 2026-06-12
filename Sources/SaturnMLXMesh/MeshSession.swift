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

    /// Live first-class weighted directed computation graph for this session (Levels 1-10).
    /// Populated during loadModel; updated via recordObservedCostFrom after generations
    /// so that future PlacementEngine / scheduler can rewrite placements (min ∑w(v)+∑w(e)).
    private var computationGraph = MeshComputationGraph()

    /// KV cache manager for per-episode compressed caches (v0.3+ EpiCache).
    /// Coordinates with episodicMemory (text) and the computationGraph (residency).
    /// Real [any KVCache] stored via internal boxes; metadata + layer budgets via LayerBudgetAllocator.
    public let kvCacheManager: KVCacheManager

    /// Placement engine / graph scheduler (Phase 3+). Uses the live computationGraph
    /// (costs + residencies) for decisions and can rewrite weights from telemetry
    /// (stub toward min ∑w(v) + ∑w(e) including episodic memory).
    public let placementEngine: PlacementEngine

    public init(
        controlPlane: ControlPlane = .local,
        policy: SessionPolicy = .appleSiliconBalanced,
        defaultSpeculativeGamma: Int? = nil,
        episodicMemory: EpisodicMemoryIndex? = nil
    ) {
        self.controlPlane = controlPlane
        self.defaultSpeculativeGamma = defaultSpeculativeGamma
        self.episodicMemory = episodicMemory ?? EpisodicMemoryIndex()
        self.kvCacheManager = KVCacheManager(index: self.episodicMemory)

        switch policy {
        case .appleSiliconBalanced:
            self.policy = PlacementPolicyKind.appleSiliconBalanced
        }
        self.placementEngine = PlacementEngine(policy: self.policy)
    }

    /// Load a model (and optionally a drafter for speculative decoding).
    public func loadModel(
        id: String,
        role: ModelRole = .primary,
        drafterId: String? = nil
    ) async throws -> MeshModel {
        let decision = placementEngine.decide(role: role, graph: computationGraph)
        let loadStart = Date()

        // Populate the session's first-class MeshComputationGraph (Phase 1).
        // This makes the multi-level vision (L1 Device, L2 Silicon+UMA~0, L4 Placement,
        // L7 Speculative edges) explicit and queryable. The graph is the schedulable artifact.
        do {
            let localDevice = Device(id: "local", kind: "AppleSilicon")
            let su = SiliconExecutionUnit(device: localDevice, unit: decision.unit)
            computationGraph.ensureDevice(localDevice)
            if let balanced = policy as? AppleSiliconBalanced {
                computationGraph.addSiliconUnit(su, initialWeight: balanced.nodeWeight(for: decision.unit))
            } else {
                computationGraph.addSiliconUnit(su)
            }
            let compKind = (role == .primary) ? "primary" : (role == .drafter ? "drafter" : "secondary")
            let comp = ModelComponent(id: id, kind: compKind)
            computationGraph.addComponent(comp, on: su)
            computationGraph.activate(comp.id)

            if let drafterId {
                // Decide placement for the drafter using the engine (now with the primary in the graph,
                // so L7 subgraph awareness in decide can influence the unit choice for the drafter).
                // Then attach the L7 speculative subgraph explicitly (drafter propose/verify edge to primary).
                // This makes the L7 structure first-class in the live graph (pluggable for future executor).
                let drafterDecision = placementEngine.decide(role: .drafter, graph: computationGraph)
                let dUnit = SiliconExecutionUnit(device: localDevice, unit: drafterDecision.unit)
                computationGraph.addL7SpeculativeDrafter(drafterID: drafterId, on: dUnit, forPrimary: id)
            }
        }

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

        // Fresh rewrite after load (graph now contains the just-placed component + any prior residencies).
        // This lets subsequent decide() calls and cost explanations see up-to-date state.
        await self.rewriteGraphUsingTelemetry()

        return model
    }

    public func telemetrySnapshot() async -> TelemetrySnapshot {
        await telemetry.snapshot()
    }

    /// Returns a snapshot of the live computation graph (devices, silicon units,
    /// components, edges, and current node weights after any observed cost feedback).
    public func currentComputationGraph() async -> MeshComputationGraph {
        computationGraph
    }

    /// Ingest new conversation turns into the episodic memory index.
    /// Call this after each user/assistant exchange for long-context support.
    public func ingest(turns: [ConversationTurn]) async {
        await episodicMemory.ingest(turns: turns)
        // After ingest we may have built (or will soon build) episode KV caches.
        // Run a rewrite so the graph immediately reflects any new residency pressure.
        await self.rewriteGraphUsingTelemetry()
    }

    /// Retrieve relevant episode text (v0.2) to augment a prompt.
    /// v0.3+: also returns EpisodeKVCache metadata (real cache available via kvCacheManager.getRealCache if prefilled).
    public func retrieveContext(for query: String, maxEpisodes: Int = 2) async -> String {
        guard let match = await episodicMemory.match(query: query) else { return "" }

        // For v0.2 just concatenate the turns from the best episode(s).
        // In a real system we would also retrieve compressed KV for that episode.
        let turnsText = match.episode.turns.map { "\($0.role): \($0.content)" }.joined(separator: "\n")
        return "Relevant prior context:\n\(turnsText)\n\n"
    }

    /// v0.3 Epi KV: retrieve both text context and the episode KV cache handle (if any).
    /// The real [any KVCache] (if donated) can be fetched via kvCacheManager.getRealCache(match.episode.id)
    /// for use as initial cache in TokenIterator (avoids re-prefill of the retrieved episode).
    public func retrieveContextWithKV(for query: String, maxEpisodes: Int = 2) async -> (text: String, kv: EpisodeKVCache?) {
        guard let match = await episodicMemory.match(query: query) else { return ("", nil) }

        let turnsText = match.episode.turns.map { "\($0.role): \($0.content)" }.joined(separator: "\n")
        let text = "Relevant prior context:\n\(turnsText)\n\n"
        let kv = await kvCacheManager.retrieveCache(for: match)
        return (text, kv)
    }

    /// Convenience: retrieve episodic context for the prompt and generate.
    /// v0.2 text-level. v0.3+: uses retrieveContextWithKV; the returned EpisodeKVCache
    /// metadata (and real cache if prefilled via kvCacheManager) enables priming the
    /// main generation cache to skip re-prefill of long retrieved episodes (EpiCache block prefill goal).
    /// Real priming hook lives in MeshModel (TokenIterator can take a pre-populated cache).
    public func generateWithMemory(
        model: MeshModel,
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7,
        speculativeGamma: Int? = nil
    ) async throws -> AsyncThrowingStream<GeneratedToken, Error> {
        let (context, _kv) = await retrieveContextWithKV(for: prompt)

        let stream: AsyncThrowingStream<GeneratedToken, Error>
        if let kv = _kv, let pre = await kvCacheManager.getPrefilledCache(for: kv.episodeID) {
            // Prime with the prefilled episode KV cache: pass only the new 'prompt' (the query),
            // the prefilled cache provides the state for the retrieved context prefix.
            // This closes the v0.3 Epi KV priming hook (avoids re-prefill of long context).
            stream = try await model.generate(
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                speculativeGamma: speculativeGamma,
                prefilledCache: pre
            )
        } else {
            let augmented = context + prompt
            stream = try await model.generate(
                prompt: augmented,
                maxTokens: maxTokens,
                temperature: temperature,
                speculativeGamma: speculativeGamma
            )
        }

        // Phase 3: after a generation that produced telemetry, let the engine rewrite the graph
        // (incorporates tps / memory pressure + current residencies into live w(v)).
        await self.rewriteGraphUsingTelemetry()
        return stream
    }

    /// Test/simulation helper: populate the graph for a model handle without
    /// triggering the real (network + weights) load path. Real loadModel always
    /// populates automatically. This keeps unit tests fast while proving the
    /// graph construction + feedback APIs.
    internal func _registerModelForGraphTest(id: String, role: ModelRole, unit: MeshExecutionUnit) async {
        let localDevice = Device(id: "local", kind: "AppleSilicon")
        let su = SiliconExecutionUnit(device: localDevice, unit: unit)
        computationGraph.ensureDevice(localDevice)
        computationGraph.addSiliconUnit(su)
        let compKind = (role == .primary) ? "primary" : (role == .drafter ? "drafter" : "secondary")
        let comp = ModelComponent(id: id, kind: compKind)
        computationGraph.addComponent(comp, on: su)
        computationGraph.activate(comp.id)
    }

    /// Blend live generation telemetry into the graph's node weights for the
    /// given model/unit. This is the Level 9 feedback mechanism that lets the
    /// (future) scheduler rewrite placements and active subgraphs.
    /// Called by tests today; in fuller runtime the session or a side-effect
    /// stage would invoke it after each GenerationInfo is recorded.
    internal func recordObservedCostFrom(info: GenerationInfo, for modelID: String, unit: MeshExecutionUnit) async {
        let siliconKey = "local:\(unit)"
        let tps = max(info.tokensPerSecond, 1.0)
        let lat = max(0.001, 1.0 / tps)          // proxy: high throughput = low latency
        let comp = max(0.5, 100.0 / tps)         // proxy for relative compute cost
        computationGraph.recordObservedCost(for: siliconKey, latency: lat, compute: comp)
        computationGraph.recordObservedCost(for: modelID, latency: lat, compute: comp)
    }

    /// Phase 3 scheduler hook: rewrite the live computation graph using current telemetry
    /// (blends tps/memory pressure into w(v); residency already updated via KV manager).
    /// This is the entry point for the "continuously rewrites the graph" behavior (L10).
    public func rewriteGraphUsingTelemetry() async {
        let snap = await telemetry.snapshot()
        var g = computationGraph
        placementEngine.rewrite(graph: &g, using: snap)
        computationGraph = g
    }

    /// Test helper: build an episode KV cache (using LayerBudgetAllocator inside manager)
    /// and mark its residency on the graph. Demonstrates v0.3 Epi KV + graph tie-in.
    internal func _buildAndRegisterEpisodeKVForTest(episode: ConversationEpisode, budget: KVCacheBudget = KVCacheBudget(maxTokens: 512), sensitivities: [Double]? = nil) async -> EpisodeKVCache {
        let kv = await kvCacheManager.buildEpisodeCache(
            episode: episode,
            budget: budget,
            layerSensitivities: sensitivities
        )
        // Mark residency on the graph (uses current "local" unit for sim; real would use decision.unit).
        computationGraph.markEpisodeResidency(episodeID: kv.episodeID, on: "local:gpu")
        return kv
    }
}

// Convenience re-exports so callers can use the types without extra imports.
public typealias SaturnMeshSession = MeshSession
public typealias SaturnMeshModel = MeshModel
