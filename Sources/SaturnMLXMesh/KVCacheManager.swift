// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// KVCacheManager.swift
// Manages episode-specific KV caches (v0.3+).
// For v0.2 this is a skeleton that can hold text-level episodes and
// future EpisodeKVCache objects (wrappers around [any KVCache]).
//
// Ties into MeshModel's KV cache reuse and the episodic index.

import Foundation

/// Budget for an episode's KV cache (tokens or bytes; advisory).
public struct KVCacheBudget: Sendable, Equatable {
    public let maxTokens: Int?
    public let maxBytes: Int?

    public init(maxTokens: Int? = nil, maxBytes: Int? = nil) {
        self.maxTokens = maxTokens
        self.maxBytes = maxBytes
    }
}

/// Placeholder for a compressed KV cache tied to one episode.
/// In v0.3 this will wrap actual [any KVCache] + metadata (layer budgets, etc.).
public struct EpisodeKVCache: Sendable, Equatable {
    public let episodeID: UUID
    public let budget: KVCacheBudget
    public let createdAt: Date
    // Future: internal compressed caches per layer, sensitivity map, etc.

    public init(episodeID: UUID, budget: KVCacheBudget) {
        self.episodeID = episodeID
        self.budget = budget
        self.createdAt = Date()
    }
}

/// Actor that owns episode KV caches and coordinates with EpisodicMemoryIndex.
/// For now (v0.2) it can store text episodes + placeholder caches.
/// v0.3 will add real block-prefill compression and per-episode KV objects.
public actor KVCacheManager {
    private var episodeCaches: [UUID: EpisodeKVCache] = [:]
    private let index: EpisodicMemoryIndex

    public init(index: EpisodicMemoryIndex) {
        self.index = index
    }

    /// Build (or rebuild) a compressed cache for an episode.
    /// v0.2: just creates a placeholder. v0.3 will do actual compression.
    public func buildEpisodeCache(
        episode: ConversationEpisode,
        fullHistory: [ConversationTurn],
        budget: KVCacheBudget
    ) async throws -> EpisodeKVCache {
        let cache = EpisodeKVCache(episodeID: episode.id, budget: budget)
        episodeCaches[episode.id] = cache
        // TODO v0.3: perform block-wise prefill on the relevant turns,
        // apply compression, allocate per-layer budgets, store real KVCache.
        return cache
    }

    /// Retrieve the best episode cache for a match (if any).
    public func retrieveCache(for match: EpisodeMatch) async throws -> EpisodeKVCache? {
        return episodeCaches[match.episode.id]
    }

    /// Update an existing episode cache with a new turn (incremental).
    public func updateCache(_ cache: EpisodeKVCache, with turn: ConversationTurn) async throws {
        // v0.2: no-op (text index will be updated separately via ingest)
        // v0.3: append to the episode's KV cache with block prefill + eviction.
        episodeCaches[cache.episodeID] = cache // placeholder
    }

    public func allCaches() async -> [EpisodeKVCache] {
        Array(episodeCaches.values)
    }
}