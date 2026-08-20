"""Shared helpers for the single-task evaluation scripts.

Used by:
  - examples/baselines/act/eval_act_single.py
  - examples/baselines/diffusion_policy/eval_dp_single.py

The implementations are lifted from eval_act_imitator.py / eval_dp_imitator.py so
the *_single scripts share identical L-level, skip/resume and summary logic
without duplicating it.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils


# =============================================================================
# L-level extraction
# =============================================================================

def extract_base_env_name(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            level, base = parts[0], parts[1]
            if level == "L3":
                if "-v" in base:
                    name_part, version_part = base.rsplit("-v", 1)
                    return f"{name_part}L3-v{version_part}"
                return f"{base}L3"
            return base
    return env_id


def extract_level(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            return parts[0]
    return "L0"


# =============================================================================
# Subprocess-safe L-level differentiation
# =============================================================================

_L_ENV_VARS = {
    "L1": "MANI_SKILL_L1",
    "L2": "MANI_SKILL_L2",
    "L3": "MANI_SKILL_L3",
}


def set_l_level(level: str):
    for env_var in _L_ENV_VARS.values():
        os.environ.pop(env_var, None)
    if level in _L_ENV_VARS:
        os.environ[_L_ENV_VARS[level]] = "1"
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    if level == "L1":
        L0_L3_utils.set_l1_enabled(True)
    elif level == "L2":
        L0_L3_utils.set_l2_enabled(True)
    elif level == "L3":
        L0_L3_utils.set_l3_enabled(True)


def clear_l_level():
    for env_var in _L_ENV_VARS.values():
        os.environ.pop(env_var, None)
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)


# =============================================================================
# JSON-based skip / resume
# =============================================================================

def load_existing_results(output_dir: Path, input_mode: str) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}
    for json_file in output_dir.glob(f"*_{input_mode}_*.json"):
        try:
            with open(json_file) as f:
                results = json.load(f)
            for r in results:
                env_id = r.get("env_id")
                if env_id and r.get("status") == "success":
                    prev = existing.get(env_id)
                    if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                        existing[env_id] = r
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return existing


def is_env_already_evaluated(
    env_id: str,
    existing_results: Dict[str, Dict],
    num_episodes: int,
) -> bool:
    prev = existing_results.get(env_id)
    if prev is None:
        return False
    if prev.get("num_episodes", 0) >= num_episodes:
        print(f"⏭️  Skipping {env_id}: already evaluated "
              f"({prev['num_episodes']} episodes, "
              f"success_once={prev.get('success_once_mean', 'N/A')})")
        return True
    print(f"⚠️  Partial eval for {env_id}: "
          f"{prev.get('num_episodes', '?')}/{num_episodes} — re-evaluating")
    return False


def save_results_incremental(all_results: List[Dict], results_json: Path):
    """Persist the full result list after each env (incremental resume)."""
    results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


# =============================================================================
# Summary table
# =============================================================================

def print_summary(all_results: List[Dict], tag: str):
    print("\n" + "=" * 80)
    print(f"📊 RESULTS SUMMARY [{tag}]")
    print("=" * 80)

    headers = ["Environment", "Level", "Success"]
    rows: List[List[str]] = []
    valid = []
    for r in all_results:
        env_id = r.get("env_id", "?")
        level  = str(r.get("level", extract_level(env_id)))
        if r.get("status") == "error":
            rows.append([env_id, level, "ERROR"])
        else:
            sr = r.get("success_once_mean")
            if sr is not None:
                valid.append(float(sr))
                rows.append([env_id, level, f"{sr:.4f}"])
            else:
                rows.append([env_id, level, str(r.get("error", "N/A"))])

    print(f"\n{headers[0]:<45} {headers[1]:<6} {headers[2]:<10}")
    print("-" * 65)
    for row in rows:
        print(f"{row[0]:<45} {row[1]:<6} {row[2]:<10}")
    print("-" * 65)
    if valid:
        mean = sum(valid) / len(valid)
        print(f"{'OVERALL AVERAGE':<45} {'ALL':<6} {mean:<10.4f}")