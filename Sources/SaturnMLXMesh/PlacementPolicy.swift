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

/// Built-in policy tuned for Apple Silicon single-node UMA devices.
/// Prefers GPU/unified for primary models, allows speculative on drafters.
public struct AppleSiliconBalanced: PlacementPolicy {
    public init() {}

    public func decide(role: ModelRole) -> PlacementDecision {
        switch role {
        case .primary:
            // Primary verifier wants maximum throughput and large KV.
            // UMA makes moving KV between CPU/GPU cheap; we still target GPU.
            return PlacementDecision(
                unit: .gpu,
                maxKVSizeHint: nil,           // let model + telemetry decide later
                allowSpeculative: true,
                notes: "Primary verifier placed on GPU/UMA for peak tokens/s"
            )

        case .drafter:
            // Drafter should be small and very fast to propose gamma tokens.
            // On many M-series chips a tiny CPU or small unified model is fine
            // and reduces contention on the main GPU context.
            return PlacementDecision(
                unit: .unified,
                maxKVSizeHint: 4096,          // drafters rarely need huge context
                allowSpeculative: true,
                notes: "Drafter prefers low-latency unified placement; speculative enabled"
            )

        case .secondary:
            // Future multi-device or multi-node path. For v0.1 we fall back to
            // same placement as primary but mark speculative disabled.
            return PlacementDecision(
                unit: .gpu,
                maxKVSizeHint: nil,
                allowSpeculative: false,
                notes: "Secondary role – speculative disabled in v0.1 single-node build"
            )
        }
    }
}

/// Convenience namespace for the policies referenced in the public MeshSession API.
public enum PlacementPolicyKind {
    /// The policy used by `MeshSession(..., policy: .appleSiliconBalanced)`
    public static let appleSiliconBalanced: any PlacementPolicy = AppleSiliconBalanced()
}
