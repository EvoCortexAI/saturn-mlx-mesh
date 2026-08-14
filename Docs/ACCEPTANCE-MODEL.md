# Acceptance model pin (mesh#1 / KF)

**Status:** Primary model identity selected. Hardware acceptance evidence **not** closed by this document alone.

## Toolchain baseline

The acceptance path is aligned with the Saturn-Node deployment baseline:

- Swift tools: **6.3**
- Xcode CI baseline: **26.6**
- Package deployment floors: **macOS 26** and **iOS 26**
- Real mesh#1 execution target: Apple Silicon macOS host used by Saturn-Node

Raising the package floors does not itself prove runtime compatibility. The target host must still complete the hardware procedure below.

## Pin

| Field | Value |
|-------|--------|
| Primary model ID | `mlx-community/Qwen3-8B-4bit` |
| Code constant | `AcceptanceModelPin.primaryModelID` |
| Role | `kf-primary-acceptance` |
| Smoke command | `swift run SaturnMLXMeshSmoke` |
| Cancellation/recovery command | `swift run SaturnMLXMeshSmoke --cancel-recovery` |
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

## Hardware procedure

Run from a clean checkout on the selected Saturn-Node Apple Silicon host. Do not paste secrets, prompts, or generated response bodies into acceptance artifacts.

```sh
git status --short --branch
git rev-parse HEAD
swift --version
xcodebuild -version
sw_vers
uname -m
swift package show-dependencies --format json
swift run SaturnMLXMeshSmoke
swift run SaturnMLXMeshSmoke --cancel-recovery
```

`SaturnMLXMeshSmoke` now exercises the real `MeshModelInferenceRuntime` contract rather than bypassing it through a lower-level direct `MeshModel` call. Its default output is metadata-only and includes model identity, model-load duration, time to first non-empty delta, generated delta count, total generation duration, finish reason, and pass/fail state.

The `--cancel-recovery` mode additionally cancels an active real generation after the first non-empty delta, verifies a cancelled terminal outcome, verifies the runtime has no remaining active request, and then requires a second real request to complete successfully using the same loaded runtime.

For a fresh-process check, exit the first invocation and run the baseline smoke again as a new process. Managed Saturn-Node service restart acceptance remains a separate Node-owned gate once the service lifecycle exists.

`--show-content` is available only for local debugging. Do not use it when collecting standard acceptance evidence.

## Evidence required to close mesh#1

Record metadata only — no prompt/response bodies in standard artifacts:

1. Host identity class (for example M4 Pro 64 GB) and OS build.
2. `saturn-mlx-mesh` commit SHA.
3. Swift and Xcode versions used for the run.
4. Resolved `mlx-swift-lm` / Hugging Face / transformers revisions from the resolution used on that host.
5. Model id + weight revision actually loaded.
6. Load time, time-to-first-token/delta, generated token/delta count, and wall duration for the fixed smoke request.
7. One explicit cancel + verified no-active-request state + subsequent successful request.
8. One fresh-process restart + subsequent successful smoke.
9. Outcome of the corresponding Saturn-Node real-runtime smoke after Node is repinned to the accepted mesh revision.

Until those facts exist in the issue trail, treat prior ad-hoc Qwen runs as **evidence**, not **acceptance**.

## Saturn-Node boundary

Saturn-Node pins the **deployment** revision of this package and the model allowlist entry. Default composition remains fail-closed (`UnavailableInferenceRuntime`). The opt-in `saturn-node --real-smoke` path may exercise real MLX only under explicit hardware testing; it does not make the service operational or authorize deployment.
