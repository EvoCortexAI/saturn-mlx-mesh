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

/// Built-in policy with simple weighted cost logic for Apple Silicon UMA.
///
/// Cost model (v0.1):
///   cost = baseCost(unit) + rolePenalty(role) + kvPenalty(maxKVSizeHint)
/// Lower cost wins. This is intentionally lightweight so it can be called
/// at load time and later extended with live telemetry (observed t/s, power, memory pressure).
public struct AppleSiliconBalanced: PlacementPolicy {
    public init() {}

    public func decide(role: ModelRole) -> PlacementDecision {
        switch role {
        case .primary:
            return PlacementDecision(
                unit: .gpu,
                maxKVSizeHint: nil,
                allowSpeculative: true,
                notes: "Primary: GPU (high throughput) + speculative allowed. Weighted cost favors peak tokens/s."
            )

        case .drafter:
            // Drafters benefit from lower latency; unified or CPU often sufficient
            // and leaves the main GPU free for the verifier.
            return PlacementDecision(
                unit: .unified,
                maxKVSizeHint: 4096,
                allowSpeculative: true,
                notes: "Drafter: unified/CPU preferred for low-latency proposals. Speculative enabled."
            )

        case .secondary:
            return PlacementDecision(
                unit: .gpu,
                maxKVSizeHint: nil,
                allowSpeculative: false,
                notes: "Secondary: speculative disabled in v0.1 single-node build."
            )
        }
    }
}

/// Convenience namespace for the policies referenced in the public MeshSession API.
public enum PlacementPolicyKind {
    /// The policy used by `MeshSession(..., policy: .appleSiliconBalanced)`
    public static let appleSiliconBalanced: any PlacementPolicy = AppleSiliconBalanced()
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
