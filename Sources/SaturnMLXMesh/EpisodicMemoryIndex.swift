// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// EpisodicMemoryIndex.swift
// Text-level episodic memory index for long conversational context (v0.2).
// Inspired by EpiCache: segment history into episodes, embed, cluster (K-means),
// retrieve relevant episodes for query.
// This is the training-free text/RAG layer. KV cache compression (block prefill,
// episode-specific caches, layer budgets) comes in v0.3+ once MLX hooks are wired.
//
// For now: pure text, in-memory, deterministic enough for tests.
// Later: integrate with KVCacheManager for compressed per-episode KV.

import Foundation

/// A single turn in a conversation.
public struct ConversationTurn: Sendable, Equatable, Codable {
    public let role: String      // "user" or "assistant" (or "system")
    public let content: String
    public let timestamp: Date?

    public init(role: String, content: String, timestamp: Date? = nil) {
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }
}

/// A clustered episode of conversation turns.
public struct ConversationEpisode: Sendable, Equatable, Identifiable, Codable {
    public let id: UUID
    public let turns: [ConversationTurn]
    public let summary: String?          // optional human-readable summary (future)
    public let embedding: [Float]        // fixed-dim embedding for similarity

    public init(id: UUID = UUID(), turns: [ConversationTurn], summary: String? = nil, embedding: [Float]) {
        self.id = id
        self.turns = turns
        self.summary = summary
        self.embedding = embedding
    }
}

/// Result of a relevance match against an episode.
public struct EpisodeMatch: Sendable, Equatable {
    public let episode: ConversationEpisode
    public let score: Float   // cosine similarity (higher is better)

    public init(episode: ConversationEpisode, score: Float) {
        self.episode = episode
        self.score = score
    }
}

/// Training-free episodic memory index.
/// Ingests turns, segments them, embeds, clusters, and allows retrieval.
/// Actor-isolated for safe concurrent use from MeshSession / MeshModel.
public actor EpisodicMemoryIndex {
    private var episodes: [ConversationEpisode] = []
    private let numEpisodes: Int
    private let windowSize: Int
    private let embeddingDim: Int
    private let similarityThreshold: Float

    public init(
        numEpisodes: Int = 8,
        windowSize: Int = 4,           // turns per segment
        embeddingDim: Int = 128,
        similarityThreshold: Float = 0.35
    ) {
        self.numEpisodes = max(1, numEpisodes)
        self.windowSize = max(1, windowSize)
        self.embeddingDim = max(8, embeddingDim)
        self.similarityThreshold = similarityThreshold
    }

    /// Ingest new turns. Segments into windows, embeds, and re-clusters.
    public func ingest(turns: [ConversationTurn]) async {
        guard !turns.isEmpty else { return }

        let segments = segment(turns: turns, windowSize: windowSize)
        for segment in segments {
            let emb = embed(turns: segment)
            let ep = ConversationEpisode(turns: segment, embedding: emb)
            episodes.append(ep)
        }

        await recluster()
    }

    /// Find the most relevant episode for a query (if above threshold).
    public func match(query: String) async -> EpisodeMatch? {
        guard !episodes.isEmpty else { return nil }

        let qEmb = embed(turns: [ConversationTurn(role: "query", content: query)])
        var best: EpisodeMatch?

        for ep in episodes {
            let sim = cosineSimilarity(a: qEmb, b: ep.embedding)
            if sim >= similarityThreshold {
                if let current = best {
                    if sim > current.score {
                        best = EpisodeMatch(episode: ep, score: sim)
                    }
                } else {
                    best = EpisodeMatch(episode: ep, score: sim)
                }
            }
        }
        return best
    }

    /// All stored episodes (for inspection / telemetry / eviction policies).
    public func episodes() async -> [ConversationEpisode] {
        episodes
    }

    /// Clear all memory (for testing or session reset).
    public func clear() {
        episodes.removeAll()
    }

    // MARK: - Internal helpers (v0.2 simplified, text only)

    private func segment(turns: [ConversationTurn], windowSize: Int) -> [[ConversationTurn]] {
        var segments: [[ConversationTurn]] = []
        var current: [ConversationTurn] = []

        for turn in turns {
            current.append(turn)
            if current.count >= windowSize {
                segments.append(current)
                current = []
            }
        }
        if !current.isEmpty {
            segments.append(current)
        }
        return segments
    }

    /// Very lightweight embedding for v0.2 (no external model).
    /// Uses character n-gram histogram hashed into fixed dimension.
    /// Good enough for topic clustering in short-to-medium conversations.
    /// Replace with real embedding model (e.g. via MLX or external) in v0.3+.
    private func embed(turns: [ConversationTurn]) -> [Float] {
        let text = turns.map { $0.content.lowercased() }.joined(separator: " ")
        guard !text.isEmpty else {
            return Array(repeating: 0.0, count: embeddingDim)
        }

        var vec = [Float](repeating: 0.0, count: embeddingDim)
        let n = 3 // trigrams

        for i in 0..<(text.count - n + 1) {
            let start = text.index(text.startIndex, offsetBy: i)
            let end = text.index(start, offsetBy: n)
            let gram = String(text[start..<end])

            let hash = abs(gram.hashValue)
            let idx = hash % embeddingDim
            vec[idx] += 1.0
        }

        // L2 normalize
        let norm = sqrt(vec.reduce(0.0) { $0 + Double($1 * $1) })
        if norm > 0 {
            for i in 0..<embeddingDim {
                vec[i] = Float(Double(vec[i]) / norm)
            }
        }
        return vec
    }

    private func cosineSimilarity(a: [Float], b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = sqrt(na) * sqrt(nb)
        return denom > 0 ? dot / denom : 0
    }

    /// Simple K-means (Lloyd) on the current episode embeddings.
    /// Re-runs on every ingest for v0.2 (small N). Later: incremental or streaming.
    private func recluster() async {
        guard episodes.count > numEpisodes else { return }

        // Initialize centroids as first N embeddings
        var centroids: [[Float]] = Array(episodes.prefix(numEpisodes).map { $0.embedding })

        for _ in 0..<10 { // max iterations, small for v0.2
            var clusters: [[Int]] = Array(repeating: [], count: numEpisodes)

            // Assign
            for (idx, ep) in episodes.enumerated() {
                var best = 0
                var bestSim: Float = -1
                for (cIdx, cent) in centroids.enumerated() {
                    let sim = cosineSimilarity(a: ep.embedding, b: cent)
                    if sim > bestSim {
                        bestSim = sim
                        best = cIdx
                    }
                }
                clusters[best].append(idx)
            }

            // Update centroids
            var newCentroids: [[Float]] = []
            for cluster in clusters {
                if cluster.isEmpty {
                    newCentroids.append(centroids[newCentroids.count]) // keep old
                    continue
                }
                var sum = Array(repeating: Float(0), count: embeddingDim)
                for i in cluster {
                    for d in 0..<embeddingDim {
                        sum[d] += episodes[i].embedding[d]
                    }
                }
                let size = Float(cluster.count)
                let avg = sum.map { $0 / size }
                newCentroids.append(avg)
            }
            centroids = newCentroids
        }

        // Re-assign episodes to final clusters and keep only the "representative"
        // (for v0.2 we just keep all, but could merge or prune here).
        // Simple: we leave episodes as-is; clustering is used only for retrieval.
        // (In a fuller version we would store one compressed rep per cluster.)
    }
}