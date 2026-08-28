# Releasing saturn-mlx-mesh

**Type:** release-procedure
**Status:** binding-after-merge; published-tag-pending for `0.2.0`
**Authority:** procedure only; merge of this file does not create a tag
**Schema:** Docs/MARKDOWN-SCHEMA.md

saturn-mlx-mesh is released as a versioned Swift package. A release is a compatibility and provenance boundary, not merely a Git tag.

Merging documentation or feature work does not itself authorize or create a release.

## Release authority

Every release requires explicit approval of the intended version and release commit. Tags are immutable after publication and must never be retargeted to another commit.

Version semantics are defined in `VERSIONING.md`.

## License and publication

- `LICENSE` must contain the unmodified official Apache License 2.0 text.
- `NOTICE` records first-party copyright and third-party / trademark carve-outs.
- Published tags are immutable provenance boundaries and must never be retargeted.
- Changelog section `0.1.0` remains under the proprietary terms present when that section was recorded. Do not tag current Apache-2.0 `main` as `0.1.0`.
- The first published semantic release is `0.2.0` and is the first Apache-2.0 release. GitHub repository visibility remains a separate publication decision.

## Preconditions

Before publishing any `0.x.y` release:

- `main` is the intended source of the release and package CI is green on the exact release commit;
- `swift package dump-package` succeeds;
- `swift test` succeeds (deterministic, no weight download);
- the Node adapter surface has been reviewed for the intended release scope;
- `CHANGELOG.md` contains a dated section for the intended version and `Unreleased` contains only later work;
- `Docs/releases/<version>.md` records adapter contract, platform/toolchain baseline, known limitations, and Node migration notes;
- `LICENSE` is the unmodified official Apache License 2.0 text and `NOTICE` is present;
- no known release-blocking correctness or security defect remains open;
- the release version and exact release commit have explicit founder approval.

A versioned release must not be published solely to replace a revision pin. The selected source state must first meet these release gates.

## Release procedure

1. Choose the version according to `VERSIONING.md`.
2. Update `CHANGELOG.md` and `Docs/releases/<version>.md` for the selected version.
3. Merge the release-preparation PR to `main` without creating a tag or GitHub release as a side effect.
4. Record the resulting exact `main` commit SHA as the release candidate.
5. Re-run or verify successful package CI for that exact commit.
6. Obtain explicit approval for the version and exact candidate SHA.
7. Create an immutable Git tag matching the semantic version, for example `0.2.0`, pointing to that exact SHA. Do not prefix with `v`.
8. Create the corresponding GitHub release. Release notes must record the exact commit SHA and may be based on `Docs/releases/<version>.md`.
9. Verify that the published tag resolves to the SHA recorded in the GitHub release.
10. Migrate Saturn-Node through a separate reviewed pull request using `.upToNextMinor(from: "0.2.0")` and a committed `Package.resolved`.

The exact release SHA is publication metadata determined after the release-preparation commit exists. Do not embed a commit's own hash into that same source commit.

## Consumer contract

Before `1.0.0`, Saturn-Node should use a bounded next-minor requirement:

```swift
.package(
    url: "https://github.com/EvoCortexAI/saturn-mlx-mesh.git",
    .upToNextMinor(from: "0.2.0")
)
```

Saturn-Node commits `Package.resolved`. CI verifies origin, resolved semantic version, and exact resolved commit SHA. A floating `branch: "main"` is never an approved Node dependency.

A narrowly approved temporary revision pin may be used only for investigation or recovery and must be removed after a versioned fix exists.

## Rollback and correction

Never retarget a published tag to repair a release. Publish a new patch or minor version instead.
