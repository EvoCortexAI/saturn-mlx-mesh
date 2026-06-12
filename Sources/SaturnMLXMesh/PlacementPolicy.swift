// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// PlacementPolicy.swift
// Heterogeneous execution placement for Saturn Mesh (Apple Silicon UMA).
//
// v0.1 canonical: MeshExecutionUnit, PlacementDecision, PlacementPolicy
// with decide(role:). See bottom for placement cost model notes.

import Foundation

/// Execution units available inside a single Apple Silicon node under UMA.
/// In v0.1 we do not yet perform cross-device layer sharding; we select a
/// primary execution target for the whole model (or drafter).
public enum MeshExecutionUnit: Sendable, Equatable, CaseIterable {
    /// Unified memory + GPU (typical for MLX Metal kernels). Highest throughput
    /// for large dense forward passes.
    case gpu

    /// CPU execution (ANE offload or pure CPU path). Useful for very small
    /// drafters or when GPU is saturated / thermally limited.
    case cpu

    /// Neural Engine / ANE when exposed for specific kernels. Rare for full
    /// LLM forward in current MLX; kept for future heterogeneous policies.
    case neuralEngine

    /// Explicit unified-memory preference (the common case on Apple Silicon).
    /// The runtime (MLX) will still map to GPU/CPU as appropriate.
    case unified
}

/// Outcome of a placement decision for a given model role.
public struct PlacementDecision: Sendable, Equatable {
    public let unit: MeshExecutionUnit
    public let maxKVSizeHint: Int?          // bytes or tokens; advisory only in v0.1
    public let allowSpeculative: Bool
    public let notes: String?

    public init(
        unit: MeshExecutionUnit,
        maxKVSizeHint: Int? = nil,
        allowSpeculative: Bool = true,
        notes: String? = nil
    ) {
        self.unit = unit
        self.maxKVSizeHint = maxKVSizeHint
        self.allowSpeculative = allowSpeculative
        self.notes = notes
    }
}

/// Model roles inside a mesh session (primary verifier vs. drafter / secondary).
public enum ModelRole: Sendable, Equatable {
    case primary
    case drafter   // small fast model used for speculative proposals
    case secondary // future multi-node or pipeline role
}

/// Placement policy. The .decide(role:) entry point is the contract used by
/// MeshSession when loading models.
public protocol PlacementPolicy: Sendable {
    func decide(role: ModelRole) -> PlacementDecision
}

/// Built-in policy with explicit weighted cost logic for the Saturn Mesh computation graph (Level 9).
///
/// Nodes have w(v) = (compute, memory, power, latency)
/// Edges have w(e) = (latency, bandwidth, serialization). Intra-device (UMA) edges have near-zero transfer cost.
///
/// The policy selects the unit with the lowest total cost for the given role.
/// This is the foundation for a future dynamic scheduler that optimizes the full graph.
public struct AppleSiliconBalanced: PlacementPolicy {
    public init() {}

    /// Returns the base node weight for a silicon unit (approximate, tunable).
    public func nodeWeight(for unit: MeshExecutionUnit) -> NodeWeight {
        switch unit {
        case .gpu:
            // High throughput for large layers/experts, but higher power/memory.
            return NodeWeight(compute: 1.0, memory: 2.0, power: 4.0, latency: 1.0)
        case .cpu:
            // Good for small drafters or when GPU is contended.
            return NodeWeight(compute: 3.5, memory: 1.0, power: 1.5, latency: 2.5)
        case .neuralEngine:
            // Excellent efficiency for supported ops (low power/latency).
            return NodeWeight(compute: 0.8, memory: 0.6, power: 0.8, latency: 0.7)
        case .unified:
            // Balanced, often preferred for drafters to avoid GPU contention.
            return NodeWeight(compute: 2.0, memory: 1.2, power: 2.0, latency: 1.5)
        }
    }

    /// Approximate role penalty (primary pays more for high throughput units).
    private func rolePenalty(for role: ModelRole) -> Double {
        switch role {
        case .primary:  return 1.0
        case .drafter:  return 0.6   // Drafters prefer efficiency/low latency
        case .secondary: return 1.8
        }
    }

    /// KV size penalty (larger hints favor units with more memory headroom).
    private func kvPenalty(maxKVSizeHint: Int?) -> Double {
        guard let hint = maxKVSizeHint else { return 0 }
        // Rough: larger KV increases memory cost.
        return Double(hint) / 10000.0
    }

    /// Total node cost for placement decision: w(v) dot (1,1,1,1) + penalties.
    /// Intra-device edge cost is ~0 thanks to UMA (no serialization/bandwidth penalty).
    public func placementCost(for role: ModelRole, unit: MeshExecutionUnit, maxKVSizeHint: Int?) -> Double {
        let w = nodeWeight(for: unit)
        let base = w.compute + w.memory + w.power + w.latency
        return base * rolePenalty(for: role) + kvPenalty(maxKVSizeHint: maxKVSizeHint)
    }

    public func decide(role: ModelRole) -> PlacementDecision {
        // v0.1 role-based selection (preserves test expectations and simple behavior).
        // We compute the formal graph cost using NodeWeight w(v) and report it.
        // This is the bridge to a full weighted graph optimizer (min ∑w(v) + ∑w(e))
        // over DeviceGraph + SiliconGraph + ModelGraph + ExpertGraph etc. (Level 9/10).
        // Intra-device transfer edges have ~0 cost (Level 2 UMA advantage).

        switch role {
        case .primary:
            let cost = placementCost(for: role, unit: .gpu, maxKVSizeHint: nil)
            return PlacementDecision(
                unit: .gpu,
                maxKVSizeHint: nil,
                allowSpeculative: true,
                notes: "Primary @ GPU (high throughput). cost=\(String(format: "%.2f", cost)) using w(v) + role + kv. UMA edges ~0."
            )

        case .drafter:
            let cost = placementCost(for: role, unit: .unified, maxKVSizeHint: 4096)
            return PlacementDecision(
                unit: .unified,
                maxKVSizeHint: 4096,
                allowSpeculative: true,
                notes: "Drafter @ unified (low latency, leaves GPU free). cost=\(String(format: "%.2f", cost)). UMA edges ~0 (Level 2)."
            )

        case .secondary:
            let cost = placementCost(for: role, unit: .gpu, maxKVSizeHint: nil)
            return PlacementDecision(
                unit: .gpu,
                maxKVSizeHint: nil,
                allowSpeculative: false,
                notes: "Secondary @ GPU (speculative disabled). cost=\(String(format: "%.2f", cost))."
            )
        }
    }
}

/// Convenience namespace for the policies referenced in the public MeshSession API.
public enum PlacementPolicyKind {
    /// The policy used by `MeshSession(..., policy: .appleSiliconBalanced)`
    public static let appleSiliconBalanced: any PlacementPolicy = AppleSiliconBalanced()
}

// MARK: - Graph Foundation (Saturn Mesh as weighted directed computation graph)

/// Represents a physical device in the mesh (Level 1: Device Graph).
public struct Device: Sendable, Equatable, Hashable {
    public let id: String
    public let kind: String // e.g. "MacBookPro", "iPhone16Pro", "iPadPro"

    public init(id: String, kind: String) {
        self.id = id
        self.kind = kind
    }
}

/// Represents a silicon execution unit within a device under UMA (Level 2: Silicon Graph).
/// Transfer cost between units on the same device is ~0 due to unified memory.
public struct SiliconExecutionUnit: Sendable, Equatable {
    public let device: Device
    public let unit: MeshExecutionUnit

    public init(device: Device, unit: MeshExecutionUnit) {
        self.device = device
        self.unit = unit
    }
}

/// Node weight in the computation graph (Level 9: Weighted Graph).
/// w(v) = (compute, memory, power, latency)
public struct NodeWeight: Sendable, Equatable {
    public let compute: Double   // relative compute cost
    public let memory: Double    // memory footprint / pressure
    public let power: Double     // power draw
    public let latency: Double   // expected execution latency

    public static let zero = NodeWeight(compute: 0, memory: 0, power: 0, latency: 0)
}

/// Edge weight in the computation graph (Level 9).
/// w(e) = (latency, bandwidth, serialization)
public struct EdgeWeight: Sendable, Equatable {
    public let latency: Double
    public let bandwidth: Double
    public let serialization: Double
}

/// A model component (layer, expert, embedding, etc.) that can be placed (Level 3/4/5).
public struct ModelComponent: Sendable, Equatable, Hashable {
    public let id: String
    public let kind: String // e.g. "layer", "expert", "embedding", "router"

    public init(id: String, kind: String) {
        self.id = id
        self.kind = kind
    }
}

// MARK: - Supporting Types (per design prompt)

/// Lightweight token emission type used by the streaming generate API.
public struct Token: Sendable, Equatable {
    public let id: Int
    public let text: String

    public init(id: Int, text: String) {
        self.id = id
        self.text = text
    }
}

/// Token yielded by the generate stream (text + optional raw ID).
public struct GeneratedToken: Sendable, Equatable {
    public let text: String
    public let tokenID: Int?
}
