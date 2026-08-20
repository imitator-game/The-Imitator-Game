# ManiSkill Dual-Arm Robot Reward Design Guide

> **Audience**: Users with imitation learning (IL) background, no prior reward engineering experience.  
> **Primary goal**: Measure *imitation quality* and *task progress* as continuous signals — not binary success.  
> **Secondary goal**: Extend the same reward for RL fine-tuning when needed.

---

## 0. Background: Why Reward Matters for IL Evaluation

Your setup:
- **Task encoder**: trained on 200 tasks (sees full diversity)
- **Policy**: trained on 60 tasks (subset)
- Three evaluation regimes:

| Split | Tasks | Description | Observed avg success |
|-------|-------|-------------|----------------------|
| **Train** | L0+L1, 60 tasks | In-distribution | ~39% |
| **Cross-Level** (Test1) | L2+L3, 60 tasks | Same task type, new difficulty | ~31% |
| **New-Skill** (Test2) | L0–L3, 80 tasks | Unseen task types | ~4.75% |

**The key problem**: Binary success misses the robot that *almost* completes a task. For New-Skill tasks, 88.75% of tasks score zero reward AND zero success — but the robot still moves toward objects. A well-designed reward captures that motion trend.

**Design principle**: reward = sum of continuous phase scores, where each phase measures one verifiable sub-behavior.

---

## 1. Core Primitives

### 1.1 Distance → Reward Mapping

Always convert distances to [0, 1] rewards. Never use raw distance as reward.

```python
import numpy as np

def tanh_reward(dist, scale=5.0):
    """Recommended default. scale=5 → r≈0.95 at dist=0.06m."""
    return 1.0 - np.tanh(scale * dist)

def exp_reward(dist, scale=10.0):
    """Sharper peak near zero. Good for precise placement."""
    return np.exp(-scale * dist)

def linear_reward(dist, max_dist=0.3):
    """Transparent but plateau-free. Good for approach phases."""
    return max(0.0, 1.0 - dist / max_dist)
```

**Recommendation**: use `tanh_reward(scale=5)` for reach/transport, `exp_reward(scale=10)` for final placement precision.

### 1.2 Grasp Continualization

Binary grasp (`is_grasping`) causes gradient cliffs. Use a 3-level proxy:

```python
def grasp_reward(tcp_pos, obj_pos, is_grasping: bool, gripper_width: float):
    """
    Level 0: TCP approaching object       → 0.0–0.33
    Level 1: Contact (narrow gripper)     → 0.33–0.67
    Level 2: Stable grasp confirmed       → 0.67–1.0
    """
    dist = np.linalg.norm(tcp_pos - obj_pos)
    proximity = tanh_reward(dist, scale=8)   # 0→1 as dist→0

    if is_grasping:
        return 0.67 + 0.33 * proximity      # confirmed grasp + stability
    elif gripper_width < 0.06:              # finger closing threshold
        return 0.33 + 0.33 * proximity      # contact stage
    else:
        return 0.33 * proximity             # approach only
```

### 1.3 Orientation Reward

For tasks requiring specific tool angle (pouring, cutting, screwing):

```python
def orientation_reward(current_quat, target_quat):
    """dot product of quaternions → angular distance."""
    dot = abs(np.dot(current_quat, target_quat))
    dot = np.clip(dot, 0, 1)
    return dot ** 2   # square to penalize large deviations more
```

### 1.4 Cyclic Motion Reward

For stir/wipe/grind tasks that require repeated motions:

```python
def cyclic_progress_reward(completed_cycles, target_cycles=3):
    """Smooth progress through N required cycles."""
    return min(1.0, completed_cycles / target_cycles)
```

---

## 2. Task Taxonomy & Standard Weight Templates

All TwoRobot tasks fall into 6 types. Match your task to a template.

### Type A — Simple Pick & Place
*Tasks: PickAppleBasket, PlaceBookBookcase, PlaceClothBasket, PlacePlateRack, PickRemoteControl, PlacePillBox, etc.*

```
R = w1·R_reach + w2·R_grasp + w3·R_transport + w4·R_place
w  = [0.15,     0.25,        0.30,            0.30]
```

```python
def compute_reward_pickplace(tcp, obj, goal, is_grasping, gripper_w):
    r_reach     = tanh_reward(dist(tcp, obj), scale=5)
    r_grasp     = grasp_reward(tcp, obj, is_grasping, gripper_w)
    r_transport = tanh_reward(dist(obj, goal), scale=5) if is_grasping else 0.0
    r_place     = exp_reward(dist(obj, goal), scale=15) if not is_grasping else 0.0
    return 0.15*r_reach + 0.25*r_grasp + 0.30*r_transport + 0.30*r_place
```

### Type B — Cyclic Tool Operation
*Tasks: StirSpoon, WipePot, CleanDesk, CleanCup, GrindFood, StirBeaker, BrushBeaker*

```
R = w1·R_reach + w2·R_grasp + w3·R_approach_target + w4·R_cycles
w  = [0.10,     0.20,        0.20,                   0.50]
```

```python
def compute_reward_cyclic(tcp, tool, target_region, is_grasping, 
                           gripper_w, completed_cycles, target_cycles=3):
    r_reach   = tanh_reward(dist(tcp, tool), scale=5)
    r_grasp   = grasp_reward(tcp, tool, is_grasping, gripper_w)
    r_approach = tanh_reward(dist(tool, target_region), scale=5) if is_grasping else 0.0
    r_cycles  = cyclic_progress_reward(completed_cycles, target_cycles)
    return 0.10*r_reach + 0.20*r_grasp + 0.20*r_approach + 0.50*r_cycles
```

**Critical**: `completed_cycles` is the **peak** value — never decrement on backswing.

### Type C — Liquid Transfer / Pour
*Tasks: PourKettle, PourCup, PourLiquidMug, PourLiquidFilter, PourKetchupFries, TransFood*

```
R = w1·R_reach + w2·R_grasp + w3·R_above_target + w4·R_tilt_angle
w  = [0.10,     0.20,        0.30,               0.40]
```

```python
def compute_reward_pour(tcp, container, target_pos, is_grasping, 
                         gripper_w, container_quat, target_quat_pour):
    r_reach  = tanh_reward(dist(tcp, container), scale=5)
    r_grasp  = grasp_reward(tcp, container, is_grasping, gripper_w)
    # Above target: XY proximity + height check
    xy_dist  = dist_xy(container, target_pos)
    height   = container[2] - target_pos[2]
    r_above  = tanh_reward(xy_dist, scale=8) * float(height > 0.05) if is_grasping else 0.0
    r_tilt   = orientation_reward(container_quat, target_quat_pour) if is_grasping else 0.0
    return 0.10*r_reach + 0.20*r_grasp + 0.30*r_above + 0.40*r_tilt
```

### Type D — Dual-Arm Coordination
*Tasks: ScanPillBottle, ScanMilkBox, PlaceFileFolder, PlaceMagazineFolder, KnifeBowlFork, PlaceShoeBox, PickAppleBananaToBaskets*

**Principle**: reward each arm independently, then sum with equal weights.

```
R = 0.5 · R_arm_left + 0.5 · R_arm_right
```

Each arm reward uses the same Type A/B template applied to that arm's sub-task.

For scan tasks specifically (approach + bring together):
```
R_scanner = w1·R_grasp_scanner + w2·R_approach_item
R_item    = w1·R_grasp_item    + w2·R_proximity_to_scanner
R_scan    = 0.5·R_scanner + 0.5·R_item + bonus·(dist_scanner_item < 0.08)
```

### Type E — Articulated Object Manipulation
*Tasks: FoldBox, LiftLidFromSkillet, OpenBox, OpenLiquidCap, FoldTowel, PutBox*

```
R = w1·R_reach + w2·R_contact + w3·R_joint_progress
w  = [0.15,     0.25,          0.60]
```

```python
def compute_reward_articulated(tcp, obj_handle, joint_pos, 
                                 joint_target, joint_range):
    r_reach   = tanh_reward(dist(tcp, obj_handle), scale=5)
    r_contact = float(tcp_contacts_object)  # binary ok here (already grasped)
    # joint_progress: 0=initial, 1=fully folded/opened
    progress  = (joint_pos - joint_initial) / (joint_target - joint_initial)
    r_joint   = np.clip(progress, 0, 1)
    return 0.15*r_reach + 0.25*r_contact + 0.60*r_joint
```

### Type F — Press / Activate
*Tasks: PressStapler, PressJuicer, PlaceScrewdriver*

```
R = w1·R_reach + w2·R_contact + w3·R_press_depth
w  = [0.20,     0.30,          0.50]
```

```python
def compute_reward_press(tcp, target_pos, press_depth, max_press_depth):
    r_reach   = tanh_reward(dist(tcp, target_pos), scale=8)
    r_contact = float(gripper_contacts_target)
    r_press   = min(1.0, press_depth / max_press_depth)
    return 0.20*r_reach + 0.30*r_contact + 0.50*r_press
```

---

## 3. Return / Place Phase Split

For pick-and-return tasks (StirSpoon: pick→stir→return; WipePot: pick→wipe→return), add a **return phase**:

```
R_total = Phase1_reward + Phase2_reward + Phase3_return_reward
weights = [0.25,           0.50,           0.25]

# Phase 3 (return): only activate after Phase 2 completes
r_return = tanh_reward(dist(obj, home_pos), scale=5) * float(cycles_complete)
```

---

## 4. L-Variant Robustness

All envs have L0/L1/L2/L3 variants. **Do not hardcode object positions**. Always use:

```python
# ✓ Correct: read from env state
obj_pos = self.spoon.pose.p   # live position each step

# ✗ Wrong: hardcoded from one variant
obj_pos = np.array([-0.12, -0.12, 0.05])
```

For L3 variants (different tool/container type), the reward function structure stays identical — only the object handle changes. Verify your `env.spoon` / `env.knife` / `env.sponge` attribute names match the current L-variant.

---

## 5. Dual-Arm Assignment

Three patterns observed across all tasks:

| Pattern | Example | Reward |
|---------|---------|--------|
| **One-active** | StirSpoon, WipePot, LiftLid | Reward only the active arm |
| **Sequential** | ScanPillBottle (right grabs scanner first, then left grabs bottle) | Phase-gate: R_arm2 activates only after R_arm1 reaches threshold |
| **Parallel** | PlaceShoeBox (both arms place simultaneously) | `R = 0.5·R_left + 0.5·R_right` |

---

## 6. Evaluation Mode: Imitation Quality Metrics

When using reward for **evaluation** (not RL training), use these four metrics:

```python
class EvalMetrics:
    peak_reward:    float   # max reward seen during episode — captures motion trend
    final_reward:   float   # reward at last step — captures task completion
    phase_reached:  int     # highest phase index robot entered (0-indexed)
    success_once:   bool    # did env.is_success() ever return True
```

**For New-Skill tasks** where success=0 is expected:
- `peak_reward > 0.2` → robot shows correct motion direction
- `peak_reward > 0.4` → robot partially executes the correct sub-task  
- `peak_reward > 0.6` → robot nearly completes

```python
def imitation_quality_score(rewards: list) -> dict:
    """Aggregate per-step rewards into eval metrics."""
    return {
        "peak":    max(rewards),
        "final":   rewards[-1],
        "mean":    np.mean(rewards),
        "phase":   get_phase_reached(rewards),
        "trend":   np.polyfit(range(len(rewards)), rewards, 1)[0]  # slope > 0 = improving
    }
```

**Why `peak` matters**: your eval data shows `success_once` and `success_at_end` can diverge (e.g., OpenBox: success_once=1.0, success_at_end=0.0). Peak reward captures the highest quality moment without penalizing unstable terminal states.

---

## 7. Diagnostic: Phase Logging

When debugging why a task fails, log rewards per-phase:

```python
# In compute_dense_reward:
info["R1_reach"]     = r_reach
info["R2_grasp"]     = r_grasp  
info["R3_transport"] = r_transport
info["R4_place"]     = r_place
info["R_total"]      = reward

# Interpret:
# R1 low → robot not approaching object (perception or motion failure)
# R1 high, R2 low → robot reaches but fails to grasp (gripper calibration)
# R2 high, R3 low → grasps but wrong trajectory (generalization gap)
# R3 high, R4 low → places at wrong location (goal recognition failure)
```

---

## 8. Case Studies

### 8.1 StirSpoon (Bug Fix Required)

**Problem in existing env**: `compute_dense_reward` rewards TCP→mug distance, but mug is static. This gives high reward for doing nothing.

**Corrected 4-phase design**:
```python
def compute_dense_reward(self, obs, action, info):
    spoon = self.spoon.pose.p
    mug   = self.mug.pose.p
    bowl  = self.bowl.pose.p
    tcp   = self.agent.tcp.pose.p
    
    # Phase 1: Reach spoon (in mug)
    r1 = tanh_reward(dist(tcp, spoon), scale=5)
    
    # Phase 2: Grasp spoon
    r2 = grasp_reward(tcp, spoon, self.is_grasping(self.spoon), self.gripper_width)
    
    # Phase 3: Move to bowl
    r3 = tanh_reward(dist_xy(spoon, bowl), scale=8) if self.is_grasping(self.spoon) else 0.0
    
    # Phase 4: Stir (count CCW cycles, reward monotonically)
    stir_progress = min(1.0, self.stir_steps / 80)   # env threshold = 80 steps
    r4 = stir_progress * float(dist_xy(spoon, bowl) < 0.08)
    
    # Phase 5 (optional): Return spoon to mug
    r5 = tanh_reward(dist(spoon, mug), scale=5) * float(stir_progress >= 1.0)
    
    reward = 0.10*r1 + 0.20*r2 + 0.15*r3 + 0.45*r4 + 0.10*r5
    return reward
```

### 8.2 WipePot / CleanDesk (Cyclic Wipe)

**Key insight**: `wipe_steps` counts contact steps with the pot/target surface. Do not reward just approach — reward surface contact + motion.

```python
r_cycles = min(1.0, self.wipe_contact_steps / (3 * AVG_STEPS_PER_PASS))
# 3 passes × ~30 steps each → threshold ≈ 90 steps
```

**Eval observation**: WipePot scores 0% success in Test1 (L2/L3) despite being a training task type (L0 at 0%). This suggests the wipe counting mechanic is sensitive to exact contact geometry — the reward should emphasize **sponge/rag XY proximity to target surface** heavily before counting contact steps.

### 8.3 ScanPillBottle / ScanMilkBox (Dual-Arm Coordination)

**Phase structure** (from solution script):
1. Left grasps scanner, Right grasps item → parallel grasp phase
2. Bring together (dist_scanner_item < 0.08) → coordination phase  
3. Return both to table → completion phase

```python
# Phase 2 is the hardest to reward — use product of both arm distances:
r_scan = tanh_reward(dist(scanner_pos, item_pos), scale=12)
# This forces both arms to move toward each other, not just one
```

**Eval observation**: ScanMilkBox generalizes well across levels (L0=1.0, L1=0.6, L2=0.8, L3=0.8). ScanPillBottle has more variance. The dual-arm coordination reward is key — weight it at 0.40+.

### 8.4 FoldBox (Articulated Joint)

**L0/L1 vs L2**: Different joint threshold (`-3.0` vs `0.5`). The reward must normalize against the correct threshold per variant.

```python
joint_target = self.get_joint_target()   # reads from env config, not hardcoded
r_joint = np.clip((joint_pos - joint_init) / (joint_target - joint_init), 0, 1)
```

**Eval observation**: FoldBox shows anomalous L0→L1 jump (success 0.20→1.00, reward 0.017→0.511). The L0 reward is near-zero despite some success, suggesting the reward function does not accumulate well for L0's larger box (requires more travel). Normalize joint displacement by joint range, not absolute value.

### 8.5 PourKettle / PourCup (Liquid Pour)

**Minimum viable reward**:
1. Grasp kettle/cup
2. Move above target cup (XY < 0.10m, height > 0.05m)
3. Tilt to pour angle (specific quaternion for each env)

```python
# Pour success check (from env):
# horizontal_dist <= 0.15 AND height > 0.12m above fries (PourKetchupFries)
# cup2 XY dist <= 0.05 AND height > 0.05m (PourCup)
r_above = float(xy_dist < threshold) * tanh_reward(height_diff, scale=-5)
# Note: negative scale because more height = better up to a point
```

**Eval observation**: PourKetchupFries has low reward despite partial success (~0.036–0.10). The existing env reward likely only gives dense reward when ketchup is already above fries — add intermediate reward for picking up the bottle.

### 8.6 PressStapler (Most Robust Task)

PressStapler is the most reward-stable task across all levels (success 80–100%, reward 0.66–0.85). Study this as a design reference:

- Simple TCP→stapler distance reward during approach
- Binary contact bonus once TCP touches stapler
- High reward simply for maintaining contact (press held = continuous reward)

This "contact = continuous reward" pattern works because the success condition (gripper links contact stapler) directly maps to the reward.

**Template for all press/activate tasks**: reward holding contact over time, not just the moment of contact.

### 8.7 PlaceMagazineFolder / PlaceFileFolder (Consistent Failure Mode)

**Pattern**: success_once=0.6–1.0 but success_at_end=0.2. Magazine placed, then slides out.

**Reward fix**: add a **stabilization bonus** active in the last 20% of episode:
```python
late_episode = (step / max_steps) > 0.80
r_stable = exp_reward(dist(item, goal), scale=20) * float(late_episode)
# Upweight precision at end of episode to discourage releasing early
```

### 8.8 New-Skill Tasks (Test2) — General Approach

For tasks the robot has never seen (GrindFood, PressJuicer, PourLiquidFilter, etc.):

**Expected robot behavior**: shows motion trends toward objects without task completion.

**Reward design goal**: capture even partial progress.

```python
# Minimal viable reward for any unseen pick-and-operate task:
r = 0.30 * r_reach_primary_object    # robot approaching anything relevant
  + 0.40 * r_grasp_primary_object    # robot grasps the main tool/item
  + 0.30 * r_approach_target         # robot moves toward where action should happen
```

Do **not** add cyclic/joint/orientation rewards for new-skill eval — those require task-specific thresholds you haven't tuned. The 3-phase minimum gives a meaningful [0, 1] score that distinguishes "no motion" from "correct approach" from "grasped and moving."

---

## 9. AI Prompt Template for Per-Task Reward Design

Use this template when asking an LLM (or another user) to design rewards for a specific task:

---

**PROMPT TEMPLATE:**

```
You are designing a dense reward function for a ManiSkill dual-arm robot task.

## Task Description
[Paste the robot_H*.json entry for this task, e.g.:]
"The right arm picks up the [OBJECT] from [LOCATION], approaches [TARGET] and 
[ACTION], then [RETURN/COMPLETE]. The left arm remains idle."

## Environment Info
- Env class: TwoRobot[TaskName]Env
- Key objects: [list from env __init__]
- Success condition: [paste from compute_success()]
- Active arm: [left / right / both]
- L-variants: L0=[description], L1=[description], L2=[description], L3=[description]

## Reward Requirements
- Output: a float in [0, 1] per step (or unnormalized to be normalized later)
- Must be CONTINUOUS — no cliffs or sudden jumps
- Must reward PARTIAL PROGRESS (reaching, grasping) even if task not completed
- Task type: [A/B/C/D/E/F from taxonomy above]

## Reference Solution
[Paste key steps from the solution script's solve() method]

## Design the reward function:
1. List phases (2–5 phases typical)
2. Assign weights (sum to 1.0)
3. Write the Python code using tanh_reward / grasp_reward / cyclic_progress_reward
4. Add one sentence explaining each phase's purpose
5. Flag any corner cases (object falls, arm collides with table, etc.)
```

---

## 10. RL Training Extension

If extending this reward for RL training (not just eval), add three components:

### 10.1 Differential Reward
Prevent reward plateau by rewarding improvement:
```python
r_rl = r_imitation + 0.1 * max(0, r_imitation - r_prev_step)
```

### 10.2 Time Penalty
Encourage efficiency (only needed for RL):
```python
r_rl -= 0.001   # small per-step penalty
```

### 10.3 Success Bonus
Sparse bonus for crossing the success threshold:
```python
if env.is_success():
    r_rl += 5.0   # one-time bonus
```

**Combined RL reward**:
```python
r_rl = r_dense_imitation * 10.0  # scale up dense component
     + 5.0 * float(is_success)   # sparse success bonus
     - 0.001                     # time penalty
```

**Note**: For imitation eval, use only `r_dense_imitation` (no time penalty, no success bonus distortion).

---

## 11. Common Pitfalls & Solutions

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Hardcoded object positions | Reward breaks on L1/L2/L3 variants | Always use `self.obj.pose.p` |
| Rewarding approach to static landmark | Reward nonzero even with no motion | Check if landmark moves; if static, skip |
| Binary grasp in reward | Gradient cliff, training instability | Use 3-level grasp proxy |
| Not peak-tracking cycles | Reward drops when arm lifts for next pass | Store `max_cycles_so_far = max(current, prev_max)` |
| Dual-arm: only reward one arm | Other arm gets ignored, learns wrong behavior | Always split reward 50/50 unless explicitly sequential |
| Return phase always active | Penalizes robot for being at target | Gate return reward with `float(phase_complete)` |
| success_once ≠ success_at_end | Task completes then object falls | Add terminal stability reward in last 20% of episode |

---

## 12. Quick Reference: Task → Template Mapping

| Task Name Pattern | Type | Key Phase | Typical success (train) |
|-------------------|------|-----------|------------------------|
| Pick*Basket/Scale/Plate | A | placement precision | 40–80% |
| Place*Rack/Bookcase/Box | A | final position XY+Z | 20–100% |
| StirSpoon, GrindFood | B | cycle count near target | 60–80% |
| WipePot, CleanDesk/Cup | B | surface contact count | 0–40% |
| PourKettle/Cup/Liquid | C | tilt angle above target | 0–40% |
| ScanPill/MilkBox | D | scanner-item proximity | 60–100% |
| FoldBox, LiftLid, OpenBox | E | joint position progress | 40–100% |
| PressStapler/Juicer | F | sustained contact | 60–100% |
| PlaceFileFolder/Magazine | A+stability | placement + hold | 20–60% |
| PlaceShoeBox (2 items) | D (parallel) | both items placed | 0–20% |

---

*Last updated: 2026-03-11 | Based on 200-task TwoRobot eval (Train L0+L1, Cross-Level L2+L3, New-Skill L0–L3)*
