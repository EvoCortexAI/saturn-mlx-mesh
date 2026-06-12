// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// MeshTelemetry.swift
// Actor for load + generation telemetry (tokens/s, speculative stats, durations).
//
// Used by MeshModel. Snapshots feed future dynamic placement decisions.

import Foundation

public struct LoadRecord: Sendable {
    public let modelID: String
    public let role: ModelRole
    public let decision: PlacementDecision
    public let loadDuration: TimeInterval
    public let timestamp: Date
}

public struct GenerationInfo: Sendable {
    public let modelID: String
    public let role: ModelRole
    public let promptTokens: Int
    public let generatedTokens: Int
    public let duration: TimeInterval
    public let tokensPerSecond: Double
    public let speculativeGamma: Int?     // nil = standard path
    public let acceptedTokens: Int?       // for speculative runs
    public let memoryPressureHint: Double? // 0...1 rough estimate if available
    public let timestamp: Date
}

public struct TelemetrySnapshot: Sendable {
    public let loads: [LoadRecord]
    public let generations: [GenerationInfo]
    public let totalGeneratedTokens: Int
    public let averageTokensPerSecond: Double?
    public let lastUpdated: Date
}

public actor MeshTelemetry {
    private var loads: [LoadRecord] = []
    private var generations: [GenerationInfo] = []

    public init() {}

    public func recordLoad(
        modelID: String,
        role: ModelRole,
        decision: PlacementDecision,
        loadDuration: TimeInterval
    ) {
        let rec = LoadRecord(
            modelID: modelID,
            role: role,
            decision: decision,
            loadDuration: loadDuration,
            timestamp: Date()
        )
        loads.append(rec)
    }

    /// Record a completed generation (standard or speculative).
    /// For speculative paths the caller supplies acceptedTokens and gamma.
    public func recordGenerationInfo(_ info: GenerationInfo) {
        generations.append(info)
    }

    public func snapshot() -> TelemetrySnapshot {
        let total = generations.reduce(0) { $0 + $1.generatedTokens }
        let avg: Double? = {
            guard !generations.isEmpty else { return nil }
            let sum = generations.reduce(0.0) { $0 + $1.tokensPerSecond }
            return sum / Double(generations.count)
        }()
        return TelemetrySnapshot(
            loads: loads,
            generations: generations,
            totalGeneratedTokens: total,
            averageTokensPerSecond: avg,
            lastUpdated: Date()
        )
    }
}
