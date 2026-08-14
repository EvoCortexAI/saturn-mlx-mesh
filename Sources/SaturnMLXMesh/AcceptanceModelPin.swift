// Copyright © 2026 EvoCortexAI S.L. All rights reserved.
//
// AcceptanceModelPin.swift
// Single primary model identity for local-inference acceptance (mesh#1) and KF.
//
// This is a product pin, not a deployment manifest. Exact weight revision,
// mlx-swift-lm commit, and measured timings are recorded after a real-hardware
// acceptance run and owned by Saturn-Node / SN01 ops notes — not invented here.

import Foundation

/// Canonical primary model identity for the single-node acceptance path.
///
/// - Primary KF / mesh#1 path: `primaryModelID`
/// - 32B-class models are **not** part of this pin; they remain optional
///   hardware demonstration material only until after the 8B path is boring
///   and repeatable.
public enum AcceptanceModelPin: Sendable {
    /// Hugging Face / mlx-community model id used by smoke and acceptance.
    public static let primaryModelID = "mlx-community/Qwen3-8B-4bit"

    /// Human-readable role label for docs and logs (no secrets).
    public static let primaryRole = "kf-primary-acceptance"

    /// Fixed non-secret prompt used by the opt-in hardware smoke.
    public static let acceptancePrompt =
        "Explain Saturn mesh inference in one short sentence."

    /// Bounded output for smoke; acceptance suites may raise this under a fixed budget.
    public static let smokeMaxTokens = 32
}
