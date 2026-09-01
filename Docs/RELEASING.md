# Releasing saturn-mlx-mesh

**Type:** release-procedure
**Status:** binding; `0.2.0` published; `0.2.x` cueing after green main CI
**Authority:** procedure only; merge of this file does not create a tag
**Schema:** Docs/MARKDOWN-SCHEMA.md

Architecture views: [`ARCHITECTURE.md`](ARCHITECTURE.md).

saturn-mlx-mesh is released as a versioned Swift package. A release is a compatibility and provenance boundary, not merely a Git tag.

There are two publication paths:

1. **Development cueing** -- automatic immutable `0.2.x` tags after green `main` CI.
2. **Formal release** -- founder-approved `0.3.0+` or `1.0.0`.

## Development cueing (`0.2.x`)

This is the path Saturn-Node uses during development.

| Rule | Value |
|---|---|
| Line | `0.2.x` |
| Trigger | Push to `main` whose CI workflow succeeded on that SHA |
| Allocator | `.github/workflows/tag-0.2-dev.yml` |
| First automatic tag | `0.2.1` |
| Increment | patch only; never skip or reuse |
| Tag target | exact CI-green `main` SHA |
| Retarget | forbidden |
| Notes file | not required |
| Founder approval | not required |
| Consumer declaration | `.upToNextMinor(from: "0.2.0")` plus committed `Package.resolved` |

Skip conditions: CI failed or cancelled; event was not a `main` push; SHA already has a `0.2.x` tag.

Cueing tags may include source-breaking adapter changes. Saturn-Node reviews `Package.resolved`.

Changelog section `0.1.0` remains proprietary history. Do not tag current `main` as `0.1.0`.

## Formal release authority

Every `0.3.0+` or `1.0.0` release requires explicit approval of the intended version and release commit. Tags are immutable after publication and must never be retargeted to another commit.

Version semantics are defined in `VERSIONING.md`.

## License and publication

- `LICENSE` must contain the unmodified official Apache License 2.0 text.
- `NOTICE` records first-party copyright and third-party / trademark carve-outs.
- Published tags are immutable provenance boundaries and must never be retargeted.
- Changelog section `0.1.0` remains under the proprietary terms present when that section was recorded.
- The first published semantic release is `0.2.0` and is the first Apache-2.0 release. GitHub repository visibility remains a separate publication decision.

## Preconditions for a formal minor or major

Before publishing any `0.3.0+` or `1.0.0` release:

- `main` is the intended source of the release and package CI is green on the exact release commit;
- `swift package dump-package` succeeds;
- `swift test` succeeds (deterministic, no weight download);
- the Node adapter surface has been reviewed for the intended release scope;
- `CHANGELOG.md` contains a dated section for the intended version;
- `Docs/releases/<version>.md` records adapter contract, platform/toolchain baseline, known limitations, and Node migration notes;
- `LICENSE` is the unmodified official Apache License 2.0 text and `NOTICE` is present;
- no known release-blocking correctness or security defect remains open;
- the release version and exact release commit have explicit founder approval.

## Formal release procedure

1. Choose the version according to `VERSIONING.md` (`0.3.0+` or `1.0.0`, never a `0.2.x` cue).
2. Update `CHANGELOG.md` and `Docs/releases/<version>.md`.
3. Merge the release-preparation PR to `main` without creating a formal tag as a side effect. The automatic cueing workflow may still allocate the next `0.2.x` tag on that merge.
4. Record the resulting exact `main` commit SHA as the release candidate.
5. Re-run or verify successful package CI for that exact commit.
6. Obtain explicit approval for the version and exact candidate SHA.
7. Create an immutable Git tag matching the semantic version, pointing to that exact SHA. Do not prefix with `v`.
8. Create the corresponding GitHub release. Release notes must record the exact commit SHA.
9. Verify that the published tag resolves to the SHA recorded in the GitHub release.
10. Migrate Saturn-Node through a separate reviewed pull request using `.upToNextMinor(from: "0.2.0")` and a committed `Package.resolved`.

## Consumer contract

Before `1.0.0`, Saturn-Node should use a bounded next-minor requirement:

```swift
.package(
    url: "https://github.com/EvoCortexAI/saturn-mlx-mesh.git",
    .upToNextMinor(from: "0.2.0")
)
```

Saturn-Node commits `Package.resolved`. CI verifies origin, resolved semantic version, and exact resolved commit SHA. A floating `branch: "main"` is never an approved Node dependency.

## Rollback and correction

Never retarget a published tag to repair a release. The next green `main` merge publishes the correction as the next `0.2.x` cue.
