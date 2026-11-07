# Astra baseline

Tool-augmented, human-video-conditioned robot control with `gpt-6-astra`, reasoning effort `xhigh`, through xfast. There is no IG parameter training or target-task fine-tuning. The model plans absolute dual-Panda end-effector targets; the host applies robot-only IK and returns new observations.

The recorded main experiment is **200 episodes, 112 native successes (56%)** under a **1,800-second cumulative planning budget per episode**. The IG seen task group scores 44/100; the held-out task group scores 68/100. These groups describe the IG split, not Astra's foundation-pretraining exposure. The November 2 reporting decision retains this 30-minute result. The incomplete extra-budget continuation is supplementary.

## Files

- `eval_astra_imitator.py`: prepare, verify, or explicitly launch the frozen workflow.
- `export_results.py`: regenerate the shared-schema CSV/JSON from immutable selected attempts or the included 200 episode records. No model calls.
- `configs/protocol_200_astra.json`: the 200 task/level/seed/demo identities and settings.
- `configs/metric_definition.json`: the verified SubSR mapping and the primary-SR reporting decision.
- `native_v4/`: byte-identical copies of the recorded policy, image bridge, action interface, native grading, and cumulative-budget implementation.
- `repro/`: recorded coordinator/deployment sources and their hashes.
- `astra/runtime.py`: materializes a separate runtime and relocates machine-local paths; it does not rewrite the native policy or tools.
- `results/astra200_30min/`: the primary CSV, summary JSON, per-episode records, and metric/provenance metadata.
- `docs/WORKFLOW.md`, `docs/METRICS.md`: interaction flow, timing and scoring definitions, comparison limits.

## Recorded settings

| Setting | Value |
|---|---|
| Model / reasoning / provider | `gpt-6-astra` / `xhigh` / xfast; no automatic model fallback |
| Evaluation set | 10 tasks × L0–L3 × environment seeds 0–4 = 200 episodes |
| Human conditioning | Same-task `human_H{id}_zed2i_ep0.mp4`; fixed across levels/seeds |
| Video input | 16 uniformly sampled, timestamped frames, plus optional frame queries |
| Robot observations | Four RGB cameras (`cam1`, `cam2`, `cam3`, `zed2i`), calibration, joint state, TCP/base poses, robot-only URDF |
| Robot RGB / delivered images | 640×480 captures; image bridge maximum edge 640; demonstration contact sheet maximum edge 1536 |
| Action interface | Absolute world-frame TCP position, unit quaternion `[w,x,y,z]`, gripper boolean for both arms |
| Per-call motion limits | ≤5 cm Euclidean translation, ≤0.35 rad rotation, 1–5 simulator controls |
| Simulator | CPU physics, SAPIEN CUDA rendering, `pd_ee_pose`, normalized dense reward; simulation pauses during planning |
| Physical budget | Maximum 500 controls; recorded control interval 0.05 seconds |
| Planning budget | 1,800 cumulative client-side seconds; not a wall-clock deadline or pure model-compute measurement |
| Primary success | Native `success_once`; native success closes the API session immediately |
| After native success | Hold the last accepted control targets without more model calls until step 500 |
| Model-declared finish | No finish tool; self-declared completion does not score or stop evaluation |
| Concurrency | Five model sessions maximum in the original coordinator; recorded GPU slots `[1,2,3,1,2]` |
| Recovery | Immutable attempts, exclusive lock, persisted ledger; no ordinary-failure retries or silent overwrites |

The model does not receive native success/reward, object truth, simulator source, evaluator traces, task annotations, other episodes or a textual task name. The extra camera/calibration/IK interface is an intentional GPT baseline setting, not the joint-action interface of Zero-WAM.

## Reproduce the frozen runtime

Run commands from the repository root. Use the existing Linux simulator environment and frozen handoff archive; no training weights are required. The archive supplies the recorded simulator source, robot assets, GPT-as-Policy app-server dependency and pinned Codex CLI 0.155.1 toolchain. Its files are hash-checked. Packaging this code on the current repository main branch does **not** mean the October 31 experiment used the current simulator revision.

Copy `configs/runtime.example.json` to a private external location and set:

- `archive_root`: the existing `gpt-baseline-native-xfast` frozen handoff archive.
- `work_root`: a new empty external path, e.g. `/external/astra_reproduction/runtime_root`. This must not be the original archive or this source checkout.
- `python_executable`: the existing simulator Python environment.
- `credentials_file`: an external JSON containing the existing xfast `api_key` and `base_url`; never commit this file.
- `human_root`: the demonstration directory matching the supplied hashes.

The recorded native client uses the proxy `http://127.0.0.1:7897`; that local proxy must be available for a live rerun. The runtime is materialized outside the repository. Archived source/asset dependencies are referenced without overwriting the original experiment.

Verify the full prepared protocol, source hashes and demonstration hashes **without calling the model**:

```bash
python -m examples.baselines.astra.eval_astra_imitator \
  --runtime-config /external/private/astra_runtime.json
```

Preview one episode's launch arguments, also without model calls:

```bash
python -m examples.baselines.astra.eval_astra_imitator \
  --runtime-config /external/private/astra_runtime.json \
  --case-id H1_L0_video_s0 --run-id astra_single_001 --gpu 1
```

An explicitly requested new evaluation adds `--execute`. Full-campaign continuation adds `--resume --execute`. Do not reuse the completed campaign's work directory for a new experiment. GPU slot availability and the account's shared concurrency quota must be checked before paid execution. The source snapshot contains historical multi-model helpers; the public entrypoint always selects the Astra profile.

## Export the results

The columns of `master_records.csv` match the supplied Zero-WAM example:

```text
policy,backbone,label,num_tasks,split,env_id,level,task,SR,SubSR,episodes,timestamp,task_base
```

There are **40 rows with 5 episodes each**. `num_tasks=0` means no IG source-task training; the ten evaluation tasks are not a C10 training regime. The summary uses `seen -> 0 -> astra` and `unseen -> 0 -> astra`. No C15/C30/C45 duplicates and no FT/Scratch groups are fabricated.

```bash
python -m examples.baselines.astra.export_results \
  --episode-records examples/baselines/astra/results/astra200_30min/episode_records.jsonl \
  --output-dir /external/astra_export
```

To rebuild from the original server artifacts, replace `--episode-records` with `--campaign-root /external/original/astra200_native_p30_c5_20261031`.

**Metric compatibility:** the supplied Zero-WAM table uses terminal `success_at_end` for SR. This Astra table intentionally uses the agreed native **`success_once`** for SR, retaining 56%. Do not merge those columns as an identical metric without noting this difference. SubSR uses the mean last recorded normalized reward, matching the reference `last_reward` mapping. A timed-out Astra trajectory ends at its recorded prefix, whereas Zero-WAM's reference runs reach 500 controls. See `docs/METRICS.md` and `export_metadata.json`.

## Validation

```bash
python -m unittest discover -s examples/baselines/astra/tests -v
```

Validation covers cohort identity, all 200 denominators including timeouts, duplicate rejection, CSV/JSON agreement, primary-versus-endpoint separation, explicit SubSR definitions and frozen-code hashes. The release validation also runs existing no-model native-stop, held-control, grading and planning-budget regressions against the materialized runtime. No new paid episode is required to export or verify the archived results.
