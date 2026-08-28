# Versioning

**Type:** versioning-policy
**Status:** binding-after-merge; published-tag-pending for `0.2.0`
**Authority:** compatibility contract only; this file is not a Git tag
**Schema:** Docs/MARKDOWN-SCHEMA.md

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

Changelog section `0.1.0` (2026-06-12) records the original private skeleton. That section is not a published Git tag on current `main` and remains under the proprietary terms present when it was written.

The first published semantic release is `0.2.0`. It is the first Apache-2.0 tagged release. The active pre-1.0 compatibility line after publication is `0.2.x`.

Hardware acceptance evidence for `mlx-community/Qwen3-8B-4bit` was gathered at provenance SHA `8ce1d6f6d6f5304f526019a5b5bcbf3f2b2f783e`. `0.2.0` is that adapter/runtime surface plus subsequent Apache-2.0 relicensing on `main`. Relicensing does not reopen the hardware gate.

## Before the first stable release

Use `0.x.y` semantic versions while the library adapter is stabilizing.

- `0.x.0` may introduce deliberate source/API changes and requires Saturn-Node migration review.
- `0.x.y` patch releases contain compatible fixes within the current minor line.
- Saturn-Node should use a bounded next-minor requirement for the approved release line, for example `.upToNextMinor(from: "0.2.0")`.
- Saturn-Node commits `Package.resolved` and reviews resolved version changes in pull requests.
- Node CI records and verifies the resolved package version, approved repository origin, and corresponding exact commit SHA.
- Direct revision dependencies are reserved for an explicitly approved temporary recovery or investigation path and must not remain the normal release contract.

The `0.2.x` contract is the stable Node adapter (`MLXInferenceRuntime` / `MeshModelInferenceRuntime` / `SimulatedMLXInferenceRuntime` / `AcceptanceModelPin`). Graph, placement, speculative, and episodic-memory research may exist in the same repository and is not part of the versioned Node contract.

Each new pre-1.0 tag is created only after its release scope is explicitly approved and all gates in `RELEASING.md` pass. This policy never creates or retargets a release by itself.

## After `1.0.0`

Stable consumers should normally use SwiftPM's next-major semantic-version requirement, for example `from: "1.0.0"`, while continuing to commit `Package.resolved`.

Breaking adapter-contract changes require a new major version. Compatible adapter features use a minor version and compatible fixes use a patch version.

## Release provenance

Every saturn-mlx-mesh release must preserve an auditable mapping:

```text
version -> immutable tag -> exact commit SHA
```

CI and release records may expose the commit SHA for provenance without requiring Node manifests to depend on that SHA directly.

## Stable release eligibility

A `1.0.0` release requires:

- Node adapter surface reviewed and frozen;
- independent package CI green;
- Saturn-Node adoption on the versioned line with committed `Package.resolved`;
- hardware acceptance evidence recorded for the pinned model path;
- research graph APIs kept out of the stable adapter contract;
- no known release-blocking correctness or security issue;
- founder approval of release scope and license.
