// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// LayerBudgetAllocator.swift
// Sensitivity-aware allocation of KV cache budget across transformer layers.
//
// Based on EpiCache observation: different layers suffer differently from
// block-prefill eviction / compression. Allocate more budget to sensitive layers.
//
// v0.2: pure math, no MLX dependency yet.
// v0.3+: will be used when building per-episode KV caches.

import Foundation

public struct LayerBudgetAllocator {
    /// Allocate a total token (or slot) budget across `numLayers` layers
    /// according to per-layer sensitivity scores.
    ///
    /// - Parameters:
    ///   - totalBudget: total KV slots/tokens available for the episode.
    ///   - layerSensitivities: array of sensitivity scores (higher = more important).
    ///     Length must == number of layers. Values should be positive.
    ///   - sharpness: exponent α that controls how peaked the allocation is.
    ///     Paper found α ≈ 2–4 worked well; α=8 was too aggressive.
    /// - Returns: array of integer budgets (one per layer), summing to totalBudget.
    public func allocate(
        totalBudget: Int,
        layerSensitivities: [Double],
        sharpness: Double = 2.0
    ) -> [Int] {
        guard totalBudget > 0, !layerSensitivities.isEmpty else {
            return Array(repeating: 0, count: max(1, layerSensitivities.count))
        }

        let numLayers = layerSensitivities.count
        let sensitivities = layerSensitivities.map { max($0, 1e-9) } // avoid zero

        // Raise to power α (sharpness)
        let powered = sensitivities.map { pow($0, sharpness) }
        let sum = powered.reduce(0.0, +)
        guard sum > 0 else {
            // uniform fallback
            let base = totalBudget / numLayers
            var result = Array(repeating: base, count: numLayers)
            result[0] += totalBudget - base * numLayers // distribute remainder
            return result
        }

        // Proportional allocation
        let allocations = powered.map { ($0 / sum) * Double(totalBudget) }

        // Convert to integers while preserving sum (largest remainder method)
        var ints = allocations.map { Int($0) }
        let remainder = totalBudget - ints.reduce(0, +)

        // Sort indices by fractional part descending
        let indicesByFrac = allocations.enumerated()
            .map { ($0.offset, $0.element - Double(Int($0.element))) }
            .sorted { $0.1 > $1.1 }
            .map { $0.0 }

        for i in 0..<min(remainder, numLayers) {
            ints[indicesByFrac[i]] += 1
        }

        // Ensure non-negative
        return ints.map { max($0, 0) }
    }
}
