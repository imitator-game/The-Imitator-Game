# GPT workflow

```mermaid
flowchart TD
    A[Select task, level, seed and fixed human ep0] --> B[Reset simulator and expose permitted observations]
    B --> C[16-frame human timeline + four RGB views + calibration and proprioception]
    C --> D[Astra xhigh plans with JavaScript and image inspection]
    D --> E[imitation_act: two absolute TCP targets and gripper states]
    E --> F[Host validates freshness, shape, finite values and motion bounds]
    F --> G[Native robot-only IK and 1–5 simulator controls]
    G --> H{Native success?}
    H -- No --> I[Return fresh RGB and measured state]
    I --> D
    H -- Yes --> J[Close model session immediately]
    J --> K[Host holds accepted targets through physical step 500]
    K --> L[Save success_once and endpoint retention separately]
    D -. 1800-second cumulative planning limit .-> M[Stop API; score budget failure if no native success]
    G -. 500 physical controls without success .-> N[Score step-limit failure]
```

## Tools and state

- `imitation_start`: called once; returns the human timeline, cameras, calibration, TCP/base/joint state, and robot-only URDF reference.
- `imitation_act`: executes one validated absolute dual-arm target for 1–5 controls. Stale observation IDs or invalid commands execute no physics.
- `imitation_reference_frame`: inspects another timestamp from this episode's fixed human demonstration without advancing the simulator.
- `imitation_inspect`: inspects already provided observation/reference images or the robot-only URDF. Evaluator truth and unrelated paths are rejected.

The Codex app-server exposes these through code-mode. Image-bearing tool results must be forwarded with `image(...)`; printing the JSON alone does not display them. The policy must inspect the fresh observations between actions. There is no learned low-level policy, object-aware motion planner, grasp macro, simulator reset or agent delegation in this workflow.

## Two clocks

The planning clock includes client-observed API queueing, network waits, retries, inference, compaction and in-memory agent planning. It excludes initialization, host tool execution, simulation/rendering, post-success holding and artifact finalization. It is not a measurement of pure server inference.

The physical clock advances only through simulator controls, up to 500. A native success may interrupt a 1–5-control action chunk. The model is closed before the remaining held controls run. Native success takes priority over later bookkeeping or a budget-boundary race.

## Batch lifecycle

The recorded coordinator admits at most five workers and records each attempt before spawning it. A first-20-case engineering/cost gate checks infrastructure and spend; its success rate does not determine whether the rest are evaluated. Completed selected cases are immutable and skipped on resume. Ordinary task failures and planning timeouts are terminal, not automatically retried. Infrastructure recovery is limited to certified pre-model startup failures with a new attempt ID.

The original 88 timeouts were later eligible for a separate 30+30-minute continuation from restored state and conversation history. That work changed concurrency to four and was incomplete at 52/88, with 23 additional successes. It is not pooled into the primary 30-minute CSV/JSON.
