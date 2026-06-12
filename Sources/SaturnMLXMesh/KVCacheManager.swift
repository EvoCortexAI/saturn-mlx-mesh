// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// KVCacheManager.swift
// Manages episode-specific KV caches (v0.3+).
// For v0.2 this is a skeleton that can hold text-level episodes and
// future EpisodeKVCache objects (wrappers around [any KVCache]).
//
// Ties into MeshModel's KV cache reuse and the episodic index.

import Foundation
import MLXLMCommon   // for KVCache protocol (used in real per-episode boxes for priming)

/// Budget for an episode's KV cache (tokens or bytes; advisory).
public struct KVCacheBudget: Sendable, Equatable {
    public let maxTokens: Int?
    public let maxBytes: Int?

    public init(maxTokens: Int? = nil, maxBytes: Int? = nil) {
        self.maxTokens = maxTokens
        self.maxBytes = maxBytes
    }
}

/// Per-episode KV cache handle (v0.3+).
/// Holds metadata + layer budgets from LayerBudgetAllocator.
/// The actual [any KVCache] lives in an internal @unchecked Sendable box inside KVCacheManager
/// (same pattern as MeshModel.KVCacheBox) because KVCache is not Sendable.
public struct EpisodeKVCache: Sendable, Equatable {
    public let episodeID: UUID
    public let budget: KVCacheBudget
    public let createdAt: Date
    public let layerBudgets: [Int]?     // per-layer allocation from allocator (v0.3+)
    public let approxTokens: Int?       // approximate filled tokens for this cache

    public init(episodeID: UUID, budget: KVCacheBudget, layerBudgets: [Int]? = nil, approxTokens: Int? = nil) {
        self.episodeID = episodeID
        self.budget = budget
        self.createdAt = Date()
        self.layerBudgets = layerBudgets
        self.approxTokens = approxTokens
    }
}

/// @unchecked Sendable wrapper for a prefilled [any KVCache] obtained from the manager.
/// Allows "donating" a real KV cache array across actor boundaries (e.g. Session -> Model)
/// for use inside a ModelContainer.perform closure (priming a TokenIterator for an episode prefix).
/// The inner array is not Sendable; the wrapper is the bridge (same pattern as KVCacheBox).
public struct PrefilledKVCache: @unchecked Sendable {
    public let cache: [any KVCache]
}

/// Actor that owns episode-specific KV caches (real MLX [any KVCache] via boxes) and
/// coordinates with EpisodicMemoryIndex + the session's MeshComputationGraph (residency).
///
/// v0.3 implementation:
/// - buildEpisodeCache can accept a prefilled cache (donated from MeshModel after block prefill
///   on the episode turns) or simulate for tests.
/// - Uses LayerBudgetAllocator (proportional + largest-remainder, sharpness α) to decide
///   per-layer KV budgets when sensitivities are known (higher sensitivity layers keep more KV).
/// - retrieve / store for priming TokenIterator in generate (avoid re-prefill for retrieved episodes).
/// - Ties residency back to the graph for memory-aware future scheduling (L6/L9/L10).
///
/// The text/RAG path (v0.2) in EpisodicMemoryIndex remains the fallback; this adds the
/// fast compressed KV path for hybrid long-context.
public actor KVCacheManager {
    private var episodeCaches: [UUID: EpisodeKVCache] = [:]

    // Real caches live here (non-Sendable [any KVCache] bridged with @unchecked Sendable box).
    private final class KVCacheBox: @unchecked Sendable {
        var cache: [any KVCache]?
        var layerBudgets: [Int] = []
    }
    private var cacheBoxes: [UUID: KVCacheBox] = [:]

    private let index: EpisodicMemoryIndex

    public init(index: EpisodicMemoryIndex) {
        self.index = index
    }

    /// Build (or rebuild) a cache for an episode.
    /// - episode: the clustered turns (from EpisodicMemoryIndex).
    /// - budget: overall limit.
    /// - layerSensitivities: optional per-layer scores (length = num layers in the model).
    ///   If provided, LayerBudgetAllocator is used for peaked allocation (α=sharpness).
    /// - prefilledCache: real [any KVCache] obtained by running a prefill forward in a
    ///   ModelContainer (the donation path from MeshModel for v0.3+ real usage).
    ///   For tests / sim this can be nil; a box is still created for metadata.
    ///
    /// Returns the metadata handle. The real cache (if any) is stored internally.
    public func buildEpisodeCache(
        episode: ConversationEpisode,
        budget: KVCacheBudget,
        layerSensitivities: [Double]? = nil,
        prefilledCache: [any KVCache]? = nil
    ) async -> EpisodeKVCache {
        let allocator = LayerBudgetAllocator()
        let numLayers = layerSensitivities?.count ?? 32 // common default; real models vary
        let sens = layerSensitivities ?? Array(repeating: 1.0, count: numLayers)

        let layerBudgets = allocator.allocate(
            totalBudget: budget.maxTokens ?? 4096,
            layerSensitivities: sens,
            sharpness: 2.5   // recommended range 2–4 per EpiCache observations
        )

        let meta = EpisodeKVCache(
            episodeID: episode.id,
            budget: budget,
            layerBudgets: layerBudgets,
            approxTokens: budget.maxTokens
        )

        episodeCaches[episode.id] = meta

        let box = KVCacheBox()
        box.cache = prefilledCache
        box.layerBudgets = layerBudgets
        cacheBoxes[episode.id] = box

        // Note: actual graph.markEpisodeResidency is performed by the caller (MeshSession)
        // after this returns, so the session can choose the current silicon unit / key.
        // This keeps the manager decoupled from placement while still enabling residency tracking.

        return meta
    }

    /// Retrieve metadata for a matched episode (if we have a KV cache for it).
    public func retrieveCache(for match: EpisodeMatch) async -> EpisodeKVCache? {
        episodeCaches[match.episode.id]
    }

    /// Update an existing episode cache incrementally with a new turn (append).
    /// v0.3 would do targeted block prefill + possible eviction using layer budgets.
    public func updateCache(_ cache: EpisodeKVCache, with turn: ConversationTurn) async {
        // Metadata update (placeholder for now; real impl would adjust approxTokens etc.)
        episodeCaches[cache.episodeID] = cache
        // Real cache append would happen via model donation path + storePrefilledCache.
    }

    public func allCaches() async -> [EpisodeKVCache] {
        Array(episodeCaches.values)
    }

    /// Retrieve the *actual* MLX KV cache array for priming (internal, for MeshModel use).
    /// Returns nil if no real prefilled cache was donated yet for this episode.
    internal func getRealCache(for episodeID: UUID) -> [any KVCache]? {
        cacheBoxes[episodeID]?.cache
    }

    /// Returns a wrapped prefilled cache for priming (Sendable holder so it can be passed
    /// from Session to Model.generate for use inside perform / TokenIterator).
    internal func getPrefilledCache(for episodeID: UUID) -> PrefilledKVCache? {
        if let c = cacheBoxes[episodeID]?.cache {
            return PrefilledKVCache(cache: c)
        }
        return nil
    }

    /// Donate / store a freshly prefilled cache (called from MeshModel after block-prefill
    /// forward pass on episode turns, inside the model's container.perform).
    internal func storePrefilledCache(for episodeID: UUID, cache: [any KVCache]) {
        if let box = cacheBoxes[episodeID] {
            box.cache = cache
        } else {
            let box = KVCacheBox()
            box.cache = cache
            cacheBoxes[episodeID] = box
        }
    }

    public func clear() {
        episodeCaches.removeAll()
        cacheBoxes.removeAll()
    }

    /// Sendable-safe query (for tests, telemetry, or diagnostics). Does not
    /// expose the non-Sendable [any KVCache] across actor boundaries.
    public func hasRealCache(for episodeID: UUID) async -> Bool {
        cacheBoxes[episodeID]?.cache != nil
    }
}