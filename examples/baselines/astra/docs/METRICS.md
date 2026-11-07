# Metrics and reference-schema mapping

## Verified reference mapping

The supplied Zero-WAM `master_records.csv` was compared with all 40 C15 seen/unseen task-level groups, using 400 per-episode `metrics.json` records in Dechao's C15 E60 evaluation. Every SR value matched the mean of `success_at_end`; every SubSR value matched the mean of `last_reward`, with maximum absolute difference 0 for both mappings. `success_once` differed from the reference SR in 7 of those 40 groups. The original export script was not located; this mapping is established by direct record reconciliation. See `reference_metric_audit.json`.

## Primary Astra result

The November 2 decision explicitly selected the completed 30-minute native-success result: 112/200 = 56%, seen task group 44/100, held-out task group 68/100. Therefore **SR in the primary Astra export is `native_success_once`**, not the terminal metric used in the supplied Zero-WAM table. This difference is recorded in the README and machine-readable export metadata.

The CSV has the same 13 columns and the summary has the same split/budget/label nesting. Structural compatibility does not assert identical metric semantics, observation/control interfaces, demonstration coverage, seeds, or time budgets.

## SubSR

SubSR is the mean `final_normalized_reward`, which is the Astra recorder's equivalent of the reference `last_reward`. All 200 selected episodes remain in its denominator. For a timeout, use the last **observed** value; do not invent rewards for unexecuted future controls. The 0-step case retains the recorder's initialized last reward of zero and is explicitly identifiable in the episode records.

The reference Zero-WAM episodes ran to 500 controls. Astra timeout prefixes generally did not. Consequently this is a budget-bounded observed-end SubSR, and the horizon difference must accompany comparison. Native dense reward is a task-specific proxy and is not a manual verdict that every semantic subtask was completed.

## Endpoint stability and extra budget

Of 112 primary native successes, 106 retained native success at physical step 500; six lost it during held controls. All 88 primary failures had no observed 500-step endpoint. Their `success_at_500` remains null in per-episode records. Endpoint coverage and retention are reported separately; they do not replace the approved 56% score.

The 30+30 continuation has 52/88 completed, 23 rescued, 36 pending at the archived supplementary snapshot. The known combined 135/200 is a lower bound under an incomplete, different execution protocol, not a final 60-minute score.

## Other comparison limits

- The 200 episodes are 10 tasks × four levels × seeds 0–4, all conditioned on same-task human ep0. Many older learned-policy runs used different seeds and rotating demonstrations.
- `num_tasks=0` denotes no IG parameter training. Do not copy this fixed cohort into C15, C30 and C45 as if they were independent training experiments.
- The GPT interface intentionally adds four views, calibration, TCP poses and URDF access, and uses end-effector targets plus IK. This is a system comparison with joint-action baselines.
- Planning time includes service waiting/retries. The failure audit recorded service anomalies in 57/88 timeouts; 15 timeouts executed at most ten controls. These cases remain in the denominator.
- The task evaluator is preserved. H20's recorded success predicate is left-arm-specific; H30's human video shows tape sealing while its simulation predicate checks a box joint. These are documented interpretation questions, not retrospectively changed scores.
