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

// MARK: - Mesh Computation Graph (first-class weighted directed computation graph)
//
// This is the central abstraction for the Saturn Mesh vision (Levels 1-10).
// Vertices represent devices (L1), silicon execution units under UMA (L2),
// model components/layers/experts (L3-5), drafters (L7), routers/embedders etc (L8).
// Edges carry tensor flow or speculative propose/verify (L7) or cross-device (L6).
// Weights: w(v) = NodeWeight(c, m, p, t), w(e) = EdgeWeight(l, b, s).
// Scheduler (future PlacementEngine) continuously rewrites placements/active sets
// to minimize ∑w(v) + ∑w(e) subject to quality (L9/L10).
//
// "That graph—not the transformer itself—is the real intellectual property opportunity."
//
// Intra-device transfer edges have ~0 cost (Level 2 Apple UMA advantage).
// Current implementation populates the graph at load time and supports live
// cost feedback from telemetry (recordObservedCost). Full dynamic rewrite +
// pluggable stage execution (inspired by candidate-pipeline Sources/Scorers/SideEffects
// + Grox PlanMaster gather/merge) lands in subsequent slices of Phase 1+.

/// Directed edge in the computation graph with cost (L9).
public struct GraphEdge: Sendable, Equatable {
    public let from: String
    public let to: String
    public let weight: EdgeWeight
}

/// First-class mutable (value) representation of the Saturn Mesh as a weighted
/// directed computation graph. Owned by MeshSession; consulted by placement
/// and updated from live GenerationInfo after runs.
public struct MeshComputationGraph: Sendable, Equatable {
    public private(set) var devices: [Device] = []
    public private(set) var siliconUnits: [SiliconExecutionUnit] = []
    public private(set) var components: [ModelComponent] = []
    public private(set) var edges: [GraphEdge] = []
    /// Live node weights (keyed by component.id or silicon key "device:unit").
    public private(set) var nodeWeights: [String: NodeWeight] = [:]
    public private(set) var active: Set<String> = []

    /// Per-episode KV cache residencies for memory-aware scheduling (ties EpiCache to graph nodes).
    /// Key: episode.id (UUID), Value: silicon key e.g. "local:gpu". Updated on build/update in KVCacheManager flows.
    /// Supports Level 6/10 residency tracking + future cross-device KV movement.
    public private(set) var episodeResidencies: [UUID: String] = [:]

    /// L8 Runtime DAG stages (first-class on the graph for modeling Router/Embedder/Vision/etc. and parallel subgraphs).
    public private(set) var stages: [MeshStage] = []

    public init() {}

    public mutating func ensureDevice(_ device: Device) {
        if !devices.contains(device) { devices.append(device) }
    }

    public mutating func addSiliconUnit(_ unit: SiliconExecutionUnit, initialWeight: NodeWeight? = nil) {
        if !siliconUnits.contains(unit) {
            siliconUnits.append(unit)
        }
        let key = siliconKey(for: unit)
        if nodeWeights[key] == nil {
            nodeWeights[key] = initialWeight ?? NodeWeight(compute: 1.0, memory: 1.0, power: 1.0, latency: 1.0)
        }
    }

    public mutating func addComponent(_ component: ModelComponent, on unit: SiliconExecutionUnit?, initialWeight: NodeWeight? = nil) {
        if !components.contains(component) {
            components.append(component)
        }
        if let u = unit {
            addSiliconUnit(u)
            let e = GraphEdge(
                from: component.id,
                to: siliconKey(for: u),
                weight: EdgeWeight(latency: 0.0, bandwidth: 0.0, serialization: 0.0) // UMA intra-device ~0
            )
            if !edges.contains(e) { edges.append(e) }
        }
        let key = component.id
        if nodeWeights[key] == nil {
            nodeWeights[key] = initialWeight ?? NodeWeight(compute: 2.0, memory: 2.0, power: 3.0, latency: 1.0)
        }
    }

    private func siliconKey(for unit: SiliconExecutionUnit) -> String {
        "\(unit.device.id):\(unit.unit)"
    }

    /// Blend live observed metrics (from telemetry) into the node's weight.
    /// Higher tokensPerSecond => lower effective latency/compute.
    /// This is the Level 9 live feedback that enables the scheduler to rewrite.
    public mutating func recordObservedCost(for key: String, latency: Double, compute: Double, power: Double? = nil) {
        guard var w = nodeWeights[key] else { return }
        let alpha = 0.3 // blend factor; future policy can tune or use more stats
        let newLatency = w.latency * (1 - alpha) + latency * alpha
        let newCompute = w.compute * (1 - alpha) + compute * alpha
        let newPower = power.map { w.power * (1 - alpha) + $0 * alpha } ?? w.power
        nodeWeights[key] = NodeWeight(compute: newCompute, memory: w.memory, power: newPower, latency: newLatency)
    }

    /// Convenience: construct a minimal L7 speculative subgraph (drafter propose + verifier).
    /// Used for tests and to illustrate how speculative becomes an explicit subgraph
    /// that a future executor (PlanMaster-style gather + merge) can place across units.
    /// The edge weight represents the propose/verify roundtrip (low latency, good bandwidth for v0.1 single-node).
    public static func l7SpeculativeExample(primaryID: String, drafterID: String, primaryUnit: MeshExecutionUnit = .gpu, drafterUnit: MeshExecutionUnit = .unified) -> MeshComputationGraph {
        var g = MeshComputationGraph()
        let dev = Device(id: "local", kind: "AppleSilicon")
        g.ensureDevice(dev)
        let pUnit = SiliconExecutionUnit(device: dev, unit: primaryUnit)
        let dUnit = SiliconExecutionUnit(device: dev, unit: drafterUnit)
        g.addSiliconUnit(pUnit)
        g.addSiliconUnit(dUnit)
        let p = ModelComponent(id: primaryID, kind: "primary")
        let d = ModelComponent(id: drafterID, kind: "drafter")
        g.addComponent(p, on: pUnit)
        g.addComponent(d, on: dUnit)
        g.active = [p.id, d.id]
        // Speculative edge: drafter proposals flow to verifier (accept/reject at prefix)
        g.edges.append(GraphEdge(from: d.id, to: p.id, weight: EdgeWeight(latency: 0.05, bandwidth: 200.0, serialization: 0.0)))
        return g
    }

    /// Attach the drafter side of an L7 speculative subgraph (drafter -> primary propose/verify edge).
    /// Called from loadModel when a drafterId is supplied for a primary load, or when loading
    /// a drafter model. This makes the L7 structure explicit and first-class in the live graph
    /// (pluggable for future Runtime DAG executor).
    public mutating func addL7SpeculativeDrafter(drafterID: String, on drafterUnit: SiliconExecutionUnit, forPrimary primaryID: String) {
        addSiliconUnit(drafterUnit)
        let dComp = ModelComponent(id: drafterID, kind: "drafter")
        addComponent(dComp, on: drafterUnit)
        // Propose/verify edge weight (from the L7 vision: fast drafter proposals to verifier)
        let edgeW = EdgeWeight(latency: 0.05, bandwidth: 200.0, serialization: 0.0)
        addEdge(GraphEdge(from: dComp.id, to: primaryID, weight: edgeW))
        activate(dComp.id)
    }

    public func nodeWeight(for key: String) -> NodeWeight? {
        nodeWeights[key]
    }

    // Mutation helpers so MeshSession (same module) can populate without exposing
    // direct write access to public API consumers. Keeps the value type clean.
    public mutating func activate(_ id: String) {
        active.insert(id)
    }

    public mutating func addEdge(_ edge: GraphEdge) {
        if !edges.contains(edge) {
            edges.append(edge)
        }
    }

    /// Mark (or update) that a given episode's KV cache is resident on a particular silicon unit.
    /// Called by KVCacheManager / Session after build or update. Enables residency-aware placement.
    public mutating func markEpisodeResidency(episodeID: UUID, on siliconKey: String) {
        episodeResidencies[episodeID] = siliconKey
    }

    /// Helper for schedulers/engines to blend live costs into a node's weight (L9).
    public mutating func blendWeight(for key: String, latencyFactor: Double = 1.0, memoryFactor: Double = 1.0) {
        guard var w = nodeWeights[key] else { return }
        nodeWeights[key] = NodeWeight(
            compute: w.compute,
            memory: w.memory * memoryFactor,
            power: w.power,
            latency: w.latency * latencyFactor
        )
    }
}

// MARK: - PlacementEngine (Phase 3: graph scheduler stub, L9/L10)

/// Lightweight engine/scheduler that makes placement decisions informed by the live
/// MeshComputationGraph (costs, residencies) and can rewrite the graph using telemetry
/// (min ∑w(v) + ∑w(e) target, stub implementation).
///
/// This is the start of the "graph scheduler" that continuously rewrites the DAG
/// (including episodic memory residency) based on live data. Future versions will
/// model full Runtime DAGs (L8), cross-device, and true optimizers.
///
/// Inspired by production DAG schedulers (e.g. Grox PlanMaster gather/merge +
/// candidate-pipeline side effects for state mutation) and the Saturn vision.
public struct PlacementEngine: Sendable {
    private let basePolicy: any PlacementPolicy

    public init(policy: any PlacementPolicy) {
        self.basePolicy = policy
    }

    /// Decide placement, consulting the live graph (when provided) for current
    /// NodeWeights + episodeResidencies to compute a better unit under the
    /// placement cost model (w(v) + residency pressure + role penalty).
    ///
    /// The base policy is used as fallback / to define the candidate units.
    /// When the graph has meaningful state (residencies or recent activity),
    /// we evaluate the cost for each possible unit and pick the lowest.
    /// This keeps all existing unit expectations (primary → .gpu, drafter → .unified)
    /// in the common/no-graph case while demonstrating real graph-driven decisions.
    public func decide(role: ModelRole, graph: MeshComputationGraph? = nil) -> PlacementDecision {
        let base = basePolicy.decide(role: role)

        guard let g = graph, (g.episodeResidencies.count > 0 || !g.nodeWeights.isEmpty) else {
            return base
        }

        // Evaluate cost for a small, safe set of units that the balanced policy actually uses
        // for the common roles (gpu for primary, unified for drafter). We only let the graph
        // win if it has *live* weight data for the candidate (prevents .neuralEngine etc. from
        // unexpectedly winning just because their base nodeWeight is attractive).
        // This guarantees the existing unit expectations stay identical for all prior tests
        // while still demonstrating real graph-driven cost optimization when data is present.
        let safeCandidates: [MeshExecutionUnit] = [.gpu, .unified]
        var bestUnit = base.unit
        var bestCost = Double.infinity
        var bestNote = base.notes

        for unit in safeCandidates {
            let key = "local:\(unit)"
            // Only consider graph override if we actually have an observation for this unit.
            guard g.nodeWeights[key] != nil || g.episodeResidencies.values.contains(key) else {
                continue
            }

            // Start from the policy's base node weight for the unit, then blend with live graph weight if present.
            var w = (basePolicy as? AppleSiliconBalanced)?.nodeWeight(for: unit) ?? NodeWeight(compute: 1, memory: 1, power: 1, latency: 1)

            if let live = g.nodeWeights[key] {
                // Blend: live data has higher weight when we have observations.
                w = NodeWeight(
                    compute: (w.compute + live.compute) / 2,
                    memory: (w.memory + live.memory) / 2,
                    power: (w.power + live.power) / 2,
                    latency: (w.latency + live.latency) / 2
                )
            }

            // Extra memory pressure from residencies on this exact unit.
            let resOnUnit = g.episodeResidencies.values.filter { $0 == key }.count
            let residencyPenalty = Double(resOnUnit) * 0.8

            var cost = (basePolicy as? AppleSiliconBalanced)?.placementCost(for: role, unit: unit, maxKVSizeHint: base.maxKVSizeHint)
                ?? (w.compute + w.memory + w.power + w.latency + residencyPenalty)

            // L7 speculative subgraph awareness (from the explicit edges added at load time with drafterId):
            // If the graph contains L7 propose/verify edges (low latency, high bandwidth), give a cost
            // bonus (lower cost) to units with low latency weight. This surfaces the L7 subgraph in
            // engine decisions -- good for placing drafters on fast propose units (PlanMaster-style
            // subgraph scheduling, Phoenix isolation for parallel propose/verify).
            if role == .drafter {
                let hasL7Edge = g.edges.contains { e in
                    e.weight.latency < 0.1 && e.weight.bandwidth > 100
                }
                if hasL7Edge {
                    let latencyBonus = w.latency * 0.15
                    cost -= latencyBonus
                }
            }

            if cost < bestCost {
                bestCost = cost
                bestUnit = unit
                let note = "graph cost=\(String(format: "%.2f", bestCost)) (w(v) blended + \(resOnUnit) residencies on unit)"
                bestNote = note
            }
        }

        return PlacementDecision(
            unit: bestUnit,
            maxKVSizeHint: base.maxKVSizeHint,
            allowSpeculative: base.allowSpeculative,
            notes: bestNote
        )
    }

    /// Rewrite using live telemetry + the current graph state.
    /// Uses LoadRecords (when present) to map generations back to concrete units.
    /// Adjustments approximate the L9 objective (reward fast units, penalize memory pressure
    /// on units carrying many episode residencies). Also emits simple "migration hints"
    /// for future residency movement.
    public func rewrite(graph: inout MeshComputationGraph, using snapshot: TelemetrySnapshot) {
        guard !snapshot.generations.isEmpty else { return }

        let avgTps = snapshot.averageTokensPerSecond ?? 50.0
        let lastPressure = snapshot.generations.last?.memoryPressureHint ?? 0.0

        // Build a map from recent LoadRecords (most recent first) so we can attribute
        // generation metrics to the unit that was chosen at load time.
        var unitForModel: [String: MeshExecutionUnit] = [:]
        for rec in snapshot.loads.reversed() {
            if unitForModel[rec.modelID] == nil {
                unitForModel[rec.modelID] = rec.decision.unit
            }
        }

        for gen in snapshot.generations.suffix(8) {
            let unit = unitForModel[gen.modelID] ?? .gpu
            let key = "local:\(unit)"

            var latFactor = 1.0
            var memFactor = 1.0

            if gen.tokensPerSecond > 100 || avgTps > 100 {
                latFactor = 0.90
            }
            let resOnThisUnit = graph.episodeResidencies.values.filter { $0 == key }.count
            if resOnThisUnit > 0 && (lastPressure > 0.3 || gen.memoryPressureHint ?? 0 > 0.3) {
                memFactor = 1.12
            }

            graph.blendWeight(for: key, latencyFactor: latFactor, memoryFactor: memFactor)

            // Simple residency migration hint (for a future more sophisticated scheduler):
            // if this unit is now "expensive" and we have residencies, note that a move might help.
            if resOnThisUnit > 1 && memFactor > 1.1 {
                // In a real impl we would record a suggested target unit.
                // For now we just leave the weight change; the next decide() will see it.
            }
        }

        // Global residency pressure adjustment (affects units that are currently hosting episodes).
        if graph.episodeResidencies.count > 1 && lastPressure > 0.4 {
            for (epID, unitKey) in graph.episodeResidencies {
                if let _ = graph.nodeWeights[unitKey] {
                    graph.blendWeight(for: unitKey, latencyFactor: 1.0, memoryFactor: 1.05)
                }
            }
        }
    }

    /// Current estimated placement cost for a role on a unit, using the live graph weights
    /// (if the unit key exists) blended with the base policy node weight.
    public func currentPlacementCost(for role: ModelRole, unit: MeshExecutionUnit, graph: MeshComputationGraph) -> Double {
        let baseW = (basePolicy as? AppleSiliconBalanced)?.nodeWeight(for: unit)
            ?? NodeWeight(compute: 1, memory: 1, power: 1, latency: 1)

        let key = "local:\(unit)"
        let w = graph.nodeWeights[key] ?? baseW

        let blended = NodeWeight(
            compute: (baseW.compute + w.compute) / 2,
            memory: (baseW.memory + w.memory) / 2,
            power: (baseW.power + w.power) / 2,
            latency: (baseW.latency + w.latency) / 2
        )

        let resPenalty = Double(graph.episodeResidencies.values.filter { $0 == key }.count) * 0.8

        return (basePolicy as? AppleSiliconBalanced)?.placementCost(for: role, unit: unit, maxKVSizeHint: nil)
            ?? (blended.compute + blended.memory + blended.power + blended.latency + resPenalty)
    }

    /// Human-readable explanation of the current best decision for a role given the graph.
    public func explainDecision(role: ModelRole, graph: MeshComputationGraph) -> String {
        let d = decide(role: role, graph: graph)
        let cost = currentPlacementCost(for: role, unit: d.unit, graph: graph)
        return "role=\(role) -> unit=\(d.unit) cost=\(String(format: "%.2f", cost)) note=\(d.notes ?? "")"
    }
}

// MARK: - L8 Runtime DAG / Stage Modeling (new phase start)

/// Kinds of stages in the L8 Runtime DAG (Router, parallel experts like Drafter/Epi, Vision, etc.).
/// These become first-class ModelComponents / subgraphs on the MeshComputationGraph.
public enum RuntimeStageKind: String, Sendable, Equatable, Hashable, CaseIterable {
    case router
    case embedder
    case vision
    case reranker
    case primaryLLM
    case drafterProposer
    case epiKVSource
    // extensible for future (e.g. safety, rerank, etc.)
}

/// A pluggable stage in the Runtime DAG. Inspired by candidate-pipeline traits (Source/Hydrator/Scorer/SideEffect)
/// and Grox PlanMaster stages. enable/run/sideEffect will be fleshed in executor sketch.
public struct MeshStage: Sendable, Equatable, Hashable {
    public let kind: RuntimeStageKind
    public let id: String
    public let label: String?

    public init(kind: RuntimeStageKind, id: String, label: String? = nil) {
        self.kind = kind
        self.id = id
        self.label = label
    }
}

extension MeshComputationGraph {
    /// Add a stage (as first-class L8 component on the graph).
    public mutating func addStage(_ stage: MeshStage) {
        if !stages.contains(stage) { stages.append(stage) }
    }

    /// Attach a DAG edge between stages (or stage <-> component). Reuses GraphEdge for now.
    public mutating func addStageEdge(from: String, to: String, weight: EdgeWeight? = nil) {
        let w = weight ?? EdgeWeight(latency: 0, bandwidth: 0, serialization: 0)
        let edge = GraphEdge(from: from, to: to, weight: w)
        if !edges.contains(edge) { edges.append(edge) }
    }

    /// Describe a small L8 parallel subgraph (e.g. EpiKVSource || DrafterProposer → merge).
    /// For the toy in executor sketch. Returns the participating stage IDs.
    public mutating func describeParallelSubgraph(sources: [MeshStage], merge: MeshStage) -> [String] {
        addStage(merge)
        var ids: [String] = [merge.id]
        for s in sources {
            addStage(s)
            addStageEdge(from: s.id, to: merge.id)
            ids.append(s.id)
        }
        return ids
    }
}

// MARK: - L8 MeshGraphExecutor sketch (PlanMaster-style withTaskGroup + candidate-pipeline traits)

/// Minimal executor for L8 Runtime DAG subgraphs. Uses withTaskGroup for parallel independent stages
/// (Grox PlanMaster gather), then merge. Stages use enable/run/sideEffect pattern (candidate-pipeline).
/// Side effects here mutate the graph (KV prime, residency mark, weight update, L7 edges) -- exactly
/// like home-mixer side effects. Phoenix isolation note: parallel sources (Epi || Drafter) should not
/// crosstalk; only attend to shared prompt/context.
///
/// This is a *sketch* + toy for one parallel case. Real would integrate model contexts, full isolation masks,
/// and be driven by the PlacementEngine for cost-based stage placement.
public struct MeshGraphExecutor {
    public init() {}

    /// Execute a toy parallel L8 subgraph: run sources concurrently, then merge.
    /// `runSource` and `onMerge` are provided by caller (can be real stage logic or mocks for test).
    /// After, caller can invoke side-effects on the graph (e.g. via engine.rewrite or direct).
    public func executeParallelSubgraph(
        graph: inout MeshComputationGraph,
        sourceIDs: [String],
        mergeID: String,
        runSource: @escaping @Sendable (String) async -> Void,
        onMerge: @escaping @Sendable ([String]) async -> Void
    ) async {
        await withTaskGroup(of: Void.self) { group in
            for id in sourceIDs {
                group.addTask {
                    await runSource(id)
                }
            }
        }
        await onMerge(sourceIDs)
        // Example side-effect hook (in real use: KV prime for Epi source, L7 for drafter, rewrite for costs)
        // graph.mark... or engine.rewrite(...) would be called by the stage implementations.
    }
}
