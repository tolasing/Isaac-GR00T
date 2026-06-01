# GR00T Isaac Sim Eval — Handoff Prompt

Share this with a Claude instance on the evaluation machine to set up and run Isaac Sim eval.

---

## Context

A GR00T N1.7 model was finetuned on 28 human hand pick-and-place demonstrations and the checkpoint is ready for evaluation in Isaac Sim using a Franka Panda robot.

---

## What Is Already Done

- Checkpoint downloaded to: `/root/Isaac-GR00T/checkpoints/groot_pick_place/checkpoint-2000/`
- Modality config: `/root/Isaac-GR00T/modality_human_hand.py`
- GR00T repo: `/root/Isaac-GR00T` (branch: `groot_training`)
- IsaacLab repo: `/root/IsaacLab`

---

## Training Details (important for action space mapping)

- **Base model:** `nvidia/GR00T-N1.7-3B`
- **Embodiment tag:** `new_embodiment`
- **Action space:** 4D relative delta `[dx, dy, dz, d_gripper]` — world-frame position deltas in metres + gripper open/close
- **Observation:** single overhead RGB camera `observation.images.top` (720×1280)
- **Prediction horizon:** 16 steps

---

## What Needs To Be Done

### Step 1 — Start the GR00T policy server

The server script is at `gr00t/eval/run_gr00t_server.py` and uses **ZMQ on port 5555** (not HTTP).

```bash
cd /root/Isaac-GR00T
source .venv/bin/activate  # or activate whichever venv has gr00t installed

python gr00t/eval/run_gr00t_server.py \
  --model-path /root/Isaac-GR00T/checkpoints/groot_pick_place/checkpoint-2000 \
  --embodiment-tag new_embodiment \
  --modality-config-path /root/Isaac-GR00T/modality_human_hand.py \
  --port 5555
```

### Step 2 — Write and run a custom Isaac Sim eval script

**Important:** There is no ready-made GR00T eval script for Isaac Sim / Franka in either repo.
- `gr00t/eval/rollout_policy.py` only supports SimplerEnv and LIBERO
- IsaacLab's GR00T scripts (`scripts/imitation_learning/locomanipulation_sdg/gr00t/`) target G1 locomotion, not Franka manipulation

You need to write a new eval script that:
1. Loads `Isaac-Lift-Cube-Franka-v0` from IsaacLab (already registered)
2. Connects to the GR00T policy server via `PolicyClient` (ZMQ)
3. Maps observations from the IsaacLab env to the format GR00T expects
4. Maps GR00T's 4D delta output `[dx, dy, dz, d_gripper]` to Franka joint/EEF commands

**Key imports:**

```python
# GR00T client
from gr00t.policy.server_client import PolicyClient

# IsaacLab env
import gymnasium as gym
import isaaclab_tasks.manager_based.manipulation.lift  # noqa registers Isaac-Lift-Cube-Franka-v0
from isaaclab_tasks.utils import parse_env_cfg
```

**PolicyClient usage:**

```python
from gr00t.policy.server_client import PolicyClient
client = PolicyClient(host="localhost", port=5555)
# client.get_action(obs_dict) -> action_dict
```

The client is ZMQ-based (not HTTP). See `gr00t/policy/server_client.py` for the full interface.

**Observation format GR00T expects** (from modality config):

```python
{
    "video.observation.images.top": np.ndarray,  # shape (T, H, W, 3), uint8
    "state.single_arm": np.ndarray,              # shape (T, 4), float32 — [x, y, z, gripper]
}
```

**Action output from GR00T:**

```python
{
    "action.single_arm": np.ndarray  # shape (horizon, 4) — [dx, dy, dz, d_gripper]
}
```

The Isaac-Lift-Cube-Franka-v0 environment uses **joint position control** by default. You will need to:
- Either use `Isaac-Lift-Cube-Franka-IK-Rel-v0` (IK relative pose control, which matches 3D delta actions better), or
- Implement an IK layer to convert 3D EEF deltas → Franka joint positions

**Recommended approach:** Use `Isaac-Lift-Cube-Franka-IK-Rel-v0` since it accepts relative EEF position deltas and aligns with the 4D training action space.

---

## Registered Isaac Lab Tasks for Reference

```
Isaac-Lift-Cube-Franka-v0           # Joint position control
Isaac-Lift-Cube-Franka-Play-v0      # Joint position control (play mode)
Isaac-Lift-Cube-Franka-IK-Abs-v0    # IK absolute pose control
Isaac-Lift-Cube-Franka-IK-Rel-v0    # IK relative pose control (best match for training)
```

---

## File Reference

| Path | Purpose |
|------|---------|
| `/root/Isaac-GR00T/checkpoints/groot_pick_place/checkpoint-2000/` | Finetuned checkpoint |
| `/root/Isaac-GR00T/modality_human_hand.py` | Modality config (needed at server startup) |
| `/root/Isaac-GR00T/gr00t/eval/run_gr00t_server.py` | Policy inference server |
| `/root/Isaac-GR00T/gr00t/policy/server_client.py` | PolicyClient and PolicyServer classes |
| `/root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/` | Franka lift env configs |
| `/root/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/rollout_policy.py` | GR00T rollout example (G1, not Franka — use as reference only) |

---

## Goal

Run 50 evaluation episodes of Franka Panda pick-and-place (cube on table) in Isaac Sim, driven by the finetuned GR00T checkpoint, and report success rate.
