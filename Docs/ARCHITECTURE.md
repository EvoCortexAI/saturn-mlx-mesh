# Architecture

**Type:** architecture
**Status:** binding-after-merge for the `0.2.x` contract diagrams
**Authority:** diagrams describe the library boundary; they do not publish a tag or make Node operational
**Schema:** Docs/MARKDOWN-SCHEMA.md

These flowcharts are the architecture views for saturn-mlx-mesh. Research graph work may exist in the tree and is out of the `0.2.x` Node contract.

## Saturn execution plane

Frontends never call this package. Node loads it in-process.

```mermaid
flowchart LR
    One[Saturn One]
    Container[Saturn Container]
    Control[Saturn-Control]
    Agent[Managed agent]
    Node[Saturn-Node]
    Mesh[saturn-mlx-mesh]
    MLX[MLX]

    One -->|client API only| Control
    Container -->|client API only| Control
    Control -->|assignment + lease| Agent
    Agent -->|workload-authenticated request| Node
    Node -->|in-process adapter| Mesh
    Mesh --> MLX

    One -. forbidden .-> Node
    Container -. forbidden .-> Node
    Mesh -. no listener / creds / orchestration .-> Control
```

## Package firewall

```mermaid
flowchart TB
    Node[Saturn-Node PEP]

    subgraph Contract["0.2.x Node contract"]
        Cap[capabilities]
        Gen[generate stream]
        Cancel[cancel requestID]
        Pin[AcceptanceModelPin]
        Sim[SimulatedMLXInferenceRuntime]
        Real[MeshModelInferenceRuntime]
    end

    subgraph Out["Out of contract"]
        Graph[Device / placement / Runtime DAG]
        Spec[Speculative / episodic research]
        Listen[Network listener]
        Auth[Workload credentials]
    end

    Node --> Cap
    Node --> Gen
    Node --> Cancel
    Node --> Pin
    Sim -->|CI, no weights| Node
    Real -->|opt-in hardware smoke| Node

    Graph -. must not become the Node API .-> Node
    Spec -. must not become the Node API .-> Node
    Listen -. belongs to Node, not this package .-> Node
    Auth -. belongs to Node, not this package .-> Node
```

## In-process inference path

```mermaid
flowchart LR
    Req[ValidatedInferenceRequest]
    Rt{Runtime}
    Sim[SimulatedMLXInferenceRuntime]
    Real[MeshModelInferenceRuntime]
    Session[MeshSession]
    Model[MeshModel]
    MLX[MLX weights]
    Stream[InferenceChunk stream]
    Stop[cancel / deadline / complete]
    Quiet[no active request]
    Next[subsequent request OK]

    Req --> Rt
    Rt -->|CI| Sim --> Stream
    Rt -->|opt-in real| Real --> Session --> Model --> MLX --> Stream
    Stream --> Stop --> Quiet --> Next
```

Default Saturn-Node composition stays fail-closed and does not construct `MeshModelInferenceRuntime`.

## Version identity

```mermaid
flowchart TB
    Policy[Docs/VERSIONING.md]
    Procedure[Docs/RELEASING.md]
    Record[Docs/releases/0.2.0.md]
    Log[CHANGELOG 0.2.0 section]
    Merge[merge release-prep PR]
    SHA[main commit SHA]
    Approve[Founder approval]
    Tag["Git tag 0.2.0"]
    GH[GitHub release notes + SHA]
    NodePin["Node .upToNextMinor from 0.2.0"]
    Resolved[Package.resolved]

    Policy --> Procedure
    Record --> Merge
    Log --> Merge
    Merge --> SHA --> Approve --> Tag --> GH
    Tag --> NodePin --> Resolved

    Log -. is not .-> Tag
    Record -. is not .-> Tag
    Merge -. does not create .-> Tag
```

## Release procedure

```mermaid
flowchart TD
    A[Choose 0.x.y] --> B[Update CHANGELOG + Docs/releases]
    B --> C[Merge prep PR to main]
    C --> D[Record exact main SHA]
    D --> E[CI green on that SHA]
    E --> F{Founder approves version + SHA?}
    F -->|no| G[Stop. No tag]
    F -->|yes| H[Tag bare semver on that SHA]
    H --> I[GitHub release records SHA]
    I --> J[Verify tag resolves to SHA]
    J --> K[Separate Node PR drops revision pin]
```

## Doc architecture

```mermaid
flowchart TB
    Schema[Docs/MARKDOWN-SCHEMA.md]
    Arch[Docs/ARCHITECTURE.md]
    Ver[Docs/VERSIONING.md]
    Rel[Docs/RELEASING.md]
    Rec[Docs/releases/x.y.z.md]
    Log[CHANGELOG.md]
    Integ[Docs/SATURN-NODE-INTEGRATION.md]
    Pin[Docs/ACCEPTANCE-MODEL.md]
    Readme[README.md]

    Schema --> Ver
    Schema --> Rel
    Schema --> Rec
    Schema --> Log
    Schema --> Arch
    Arch --> Readme
    Arch --> Integ
    Pin --> Rec
    Ver --> Rel --> Rec
    Rec --> Log
```
