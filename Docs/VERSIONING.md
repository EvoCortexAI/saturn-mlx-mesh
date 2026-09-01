# Versioning

**Type:** versioning-policy
**Status:** binding; `0.2.0` published; `0.2.x` cueing after green main CI
**Authority:** compatibility contract only; this file is not a Git tag
**Schema:** Docs/MARKDOWN-SCHEMA.md

Architecture views: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Principle

saturn-mlx-mesh uses semantic versions as the consumer-facing dependency contract for Saturn-Node.

A released version identifies an immutable Git tag, and that tag resolves to one exact commit SHA. Saturn-Node declares a bounded semantic version requirement and commits `Package.resolved` so the resolved source remains reproducible. Raw commit SHAs remain provenance and CI-verification data; they are not the normal Node dependency interface.

```text
semantic version = compatibility contract
Git tag          = immutable release identity
commit SHA       = source provenance
Package.resolved = exact consumer resolution
```

Do not use a floating branch as a Node dependency. Do not retarget a released version tag.

The operational release procedure is defined in `RELEASING.md`.

## Current release line

Changelog section `0.1.0` (2026-06-12) records the original private skeleton. That section is not a published Git tag on current `main` and remains under the proprietary terms present when it was written. Do not tag current Apache-2.0 `main` as `0.1.0`.

The first published semantic release is `0.2.0`. It is the first Apache-2.0 tagged release. The active pre-1.0 compatibility and development-cueing line is `0.2.x`.

Hardware acceptance evidence for `mlx-community/Qwen3-8B-4bit` was gathered at provenance SHA `8ce1d6f6d6f5304f526019a5b5bcbf3f2b2f783e`. `0.2.0` is that adapter/runtime surface plus subsequent Apache-2.0 relicensing on `main`. Relicensing does not reopen the hardware gate. Cueing tags do not reopen it either.

## Development cueing (`0.2.x`)

Every merge to `main` that has a green package CI run on that exact commit is assigned the next unused `0.2.x` tag.

Rules:

- Start after the immutable `0.2.0` tag. The first automatic cue is `0.2.1`.
- Increment only the patch component: `0.2.1`, `0.2.2`, ... Never skip, reuse, or retarget a patch.
- Tag the exact CI-green `main` SHA. Do not tag a merge commit that failed CI.
- A commit that already carries a `0.2.x` tag is left unchanged.
- Cueing tags are Apache-2.0.
- Cueing tags may include source-breaking adapter changes. Saturn-Node stays on `.upToNextMinor(from: "0.2.0")` and reviews `Package.resolved`.
- Cueing tags do not require a `Docs/releases/<version>.md` file or founder approval.
- `0.3.0` and later minors, and `1.0.0`, remain founder-gated formal releases under `RELEASING.md`.

After each new cue, Saturn-Node runs `swift package update saturn-mlx-mesh` and commits `Package.resolved`. Do not pin `main`.

The `0.2.x` contract is the stable Node adapter (`MLXInferenceRuntime` / `MeshModelInferenceRuntime` / `SimulatedMLXInferenceRuntime` / `AcceptanceModelPin`). Graph, placement, speculative, and episodic-memory research may exist in the same repository and is not part of the versioned Node contract.

## Before the first stable release

Use `0.x.y` semantic versions while the library adapter is stabilizing.

- The development cueing line is `0.2.x` as defined above.
- A later `0.x.0` minor is a formal compatibility reset and requires Saturn-Node migration review plus the gates in `RELEASING.md`.
- Saturn-Node should use a bounded next-minor requirement, for example `.upToNextMinor(from: "0.2.0")`.
- Direct revision dependencies are reserved for an explicitly approved temporary recovery or investigation path and must not remain the normal release contract.

## After `1.0.0`

Stable consumers should normally use SwiftPM's next-major semantic-version requirement, for example `from: "1.0.0"`, while continuing to commit `Package.resolved`.

Breaking adapter-contract changes require a new major version. Compatible adapter features use a minor version and compatible fixes use a patch version. Automatic `0.2.x` cueing does not continue after `1.0.0`.

## Release provenance

Every saturn-mlx-mesh release must preserve an auditable mapping:

```text
version -> immutable tag -> exact commit SHA
```

## Stable release eligibility

A `1.0.0` release requires:

- Node adapter surface reviewed and frozen;
- independent package CI green;
- Saturn-Node adoption on the versioned line with committed `Package.resolved`;
- hardware acceptance evidence recorded for the pinned model path;
- research graph APIs kept out of the stable adapter contract;
- no known release-blocking correctness or security issue;
- founder approval of release scope and license.
