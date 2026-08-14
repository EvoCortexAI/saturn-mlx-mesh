# Acceptance model pin (mesh#1 / KF)

**Status:** Primary model identity selected. Hardware acceptance evidence **not** closed by this document alone.

## Pin

| Field | Value |
|-------|--------|
| Primary model ID | `mlx-community/Qwen3-8B-4bit` |
| Code constant | `AcceptanceModelPin.primaryModelID` |
| Role | `kf-primary-acceptance` |
| Smoke command | `swift run SaturnMLXMeshSmoke` |
| Smoke prompt | `AcceptanceModelPin.acceptancePrompt` |
| Smoke max tokens | `32` |

Source of truth in code: `Sources/SaturnMLXMesh/AcceptanceModelPin.swift`.

## What this pin is

- The **one** model identity the library smoke and KF primary path must use.
- The allowlist candidate Saturn-Node should reference in manifests for the single-node gate.

## What this pin is not

- Not a closed mesh#1 ticket. Closing requires measured evidence on target hardware.
- Not a deployment authorization for SN01, launchd, firewall, or production credentials.
- Not a commitment to multi-model product support before KF.
- **Not** Qwen 32B. 32B remains optional second-slide / backup proof only after 8B is repeatable.

## Evidence required to close mesh#1

Record (metadata only — no prompt/response bodies in standard artifacts):

1. Host identity class (e.g. M4 Pro 64 GB) and OS build.
2. `saturn-mlx-mesh` commit SHA.
3. Resolved `mlx-swift-lm` / Hugging Face / transformers revisions from the resolution used on that host.
4. Model id + weight revision actually loaded.
5. Load time, time-to-first-token, tokens generated, wall duration for the fixed smoke prompt.
6. One explicit cancel + subsequent successful request.
7. One process restart + subsequent successful smoke.

Until those numbers exist in the issue trail, treat prior ad-hoc Qwen runs as **evidence**, not **acceptance**.

## Saturn-Node boundary

Saturn-Node pins the **deployment** revision of this package and the model allowlist entry. Default composition remains fail-closed (`UnavailableInferenceRuntime`). Opt-in real runtime selection is a separate Founder-gated step after mesh#1 evidence is recorded.
