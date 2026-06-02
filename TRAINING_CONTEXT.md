# GR00T Training Context — L40 Machine Summary
> Share this with Claude on the L40 machine before starting re-training (v2).

---

## What Was Done (v1 — original training)

### 1. Environment Setup
- OS: Ubuntu 22.04, user: `satish`
- Added `satish` to sudo group
- Installed Docker (v29.5.2) and added satish to docker group
- Built the GR00T Docker image: `docker build -f docker/Dockerfile.groot -t gr00t .`
- Repo: `https://github.com/tolasing/Isaac-GR00T` (branch: `groot_training`)

### 2. Dataset
- Received 28 human hand pick-and-place demo episodes via `scp` from local machine
- Moved dataset to: `/home/satish/Isaac-GR00T/lerobot_dataset/`
- Fixed `meta/info.json`: changed `chunk_index` → `episode_chunk` in path patterns (required by GR00T loader)
- Created `meta/modality.json` (required by GR00T dataset loader)

**Dataset stats:**
- 28 episodes, 8423 frames, 25fps
- State: `[wrist_x, wrist_y, wrist_z, gripper]` — 4D, MediaPipe world coords (metres)
- Action: `[d_wrist_x, d_wrist_y, d_wrist_z, d_gripper]` — 4D relative delta
- Video: `observation.images.top` (720×1280, overhead camera)
- Scene: white table, 30×20×20cm cardboard box (target), 8cm cube

### 3. v1 Modality Config
- Embodiment tag: `new_embodiment`
- State keys: `["single_arm", "gripper"]` — 3D wrist position + gripper
- Action: `NON_EEF`, `RELATIVE`, 16-step horizon

### 4. v1 Finetuning Command
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/satish/Isaac-GR00T/lerobot_dataset:/data/lerobot_dataset \
  -v /home/satish/Isaac-GR00T/checkpoints:/data/checkpoints \
  -v /home/satish/Isaac-GR00T/modality_human_hand.py:/data/modality_human_hand.py \
  -e USE_WANDB=0 \
  -e HF_TOKEN=<your_hf_token> \
  gr00t \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/lerobot_dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path /data/modality_human_hand.py \
    --output-dir /data/checkpoints/groot_pick_place
```

**Training details:**
- Base model: `nvidia/GR00T-N1.7-3B` (~3B params)
- Trainable parameters: 1.62B (51.54%) — projector + diffusion head only
- Stopped at step ~3000
- Final loss: ~0.058–0.069
- Best checkpoint: `checkpoint-2000` → uploaded to `https://huggingface.co/tolasing/groot-pick-place`

**HuggingFace access required (gated models):**
- `nvidia/GR00T-N1.7-3B` — accept at huggingface.co/nvidia/GR00T-N1.7-3B
- `nvidia/Cosmos-Reason2-2B` — accept at huggingface.co/nvidia/Cosmos-Reason2-2B

---

## What Was Found During Isaac Sim Eval (L4 Machine)

Eval was run using:
- Two Docker containers: `gr00t:latest` (policy server) and `isaac-lab-base-gui` (Isaac Sim)
- Custom scene: white table with legs, Franka Panda on table surface, 30×20×20cm cardboard box, 8cm cube, 45° overhead camera
- 15 episodes run → ~7% success rate (1–2 successes)

**Root cause of low success: state coordinate mismatch**

The v1 modality config included `single_arm` (3D wrist position) in the state. These values in training were MediaPipe world coordinates — tiny values centred near zero (mean `[0.031, 0.075, 0.002]`, std `[0.016, 0.017, 0.027]`).

At inference the Franka EEF position in Isaac Lab world frame was sent instead (values like `[0.5, 0.0, 1.25]`), which is **30–46 standard deviations** outside the training distribution after internal normalisation. The model received nonsensical state input.

**Fix: remove `single_arm` from state, keep only `gripper`.**
The gripper signal (open/close) is meaningful and mappable between human hand and robot.

---

## Re-Training (v2) — What To Do on the L40

### What Changed
Only one change to `modality_human_hand.py`:

```python
# v1 (broken)
"state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),

# v2 (fixed)
"state": ModalityConfig(delta_indices=[0], modality_keys=["gripper"]),
```

The dataset, videos, and action space are **unchanged**. Same 28 episodes.

### Step 1 — Copy the updated modality config to the L40
The updated `modality_human_hand.py` is in the repo at branch `groot_training`. Pull latest or copy the file manually. Confirm it has `modality_keys=["gripper"]` in the state block.

### Step 2 — Run re-training
```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/satish/Isaac-GR00T/lerobot_dataset:/data/lerobot_dataset \
  -v /home/satish/Isaac-GR00T/checkpoints:/data/checkpoints \
  -v /home/satish/Isaac-GR00T/modality_human_hand.py:/data/modality_human_hand.py \
  -e USE_WANDB=0 \
  -e HF_TOKEN=<your_hf_token> \
  gr00t \
  bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/lerobot_dataset \
    --embodiment-tag new_embodiment \
    --modality-config-path /data/modality_human_hand.py \
    --output-dir /data/checkpoints/groot_pick_place_v2
```

### Step 3 — Upload best checkpoint
Upload `checkpoint-2000` (or whichever has lowest loss) to HuggingFace:
```bash
huggingface-cli upload tolasing/groot-pick-place-v2 \
  /home/satish/Isaac-GR00T/checkpoints/groot_pick_place_v2/checkpoint-2000 .
```

---

## Isaac Sim Eval Setup (L4 Machine) — Already Working

### Infrastructure
- Policy server: `gr00t:latest` Docker image
- Isaac Sim: `isaac-lab-base-gui` Docker container (host network mode, so ZMQ port 5555 is shared)
- Eval script: `/root/groot/scripts/eval_groot_franka.py` (mounted at `/workspace/isaaclab/scripts/` inside container)
- Checkpoint location: `/root/Isaac-GR00T/checkpoints/groot_pick_place/checkpoint-2000/`

### Terminal 1 — Start policy server (gr00t container)
```bash
docker run --rm --gpus all --ipc=host --network host \
  -e HF_TOKEN=<your_hf_token> \
  -v /root/Isaac-GR00T/checkpoints:/checkpoints \
  -v /root/Isaac-GR00T/modality_human_hand.py:/workspace/modality_human_hand.py \
  gr00t:latest \
  python /workspace/gr00t/eval/run_gr00t_server.py \
    --model-path /checkpoints/groot_pick_place_v2/checkpoint-2000 \
    --embodiment-tag new_embodiment \
    --modality-config-path /workspace/modality_human_hand.py \
    --port 5555
```

### Terminal 2 — Run Isaac Sim eval
```bash
docker exec -it isaac-lab-base-gui bash
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/scripts/eval_groot_franka.py \
  --headless --enable_cameras --num_episodes 50
```

### Observation format sent to server (v2)
```python
obs = {
    "video": {"top": np.array(...)},          # (1, T=1, 720, 1280, 3) uint8
    "state": {
        "gripper": np.array([[[ grip_norm ]]]) # (1, T=1, 1) float32
                                               # normalised: 0.013=closed, 0.5=open
    },
    "language": {
        "annotation.human.task_description": [["pick up the cube and place it inside the cardboard box"]]
    },
}
# result[0] = {"single_arm": (1,16,3), "gripper": (1,16,1)}
```

---

## Key Files

| File | Purpose |
|------|---------|
| `modality_human_hand.py` | GR00T modality config (v2: state=gripper only) |
| `lerobot_dataset/meta/modality.json` | Dataset-side modality config |
| `lerobot_dataset/meta/info.json` | Dataset metadata |
| `examples/finetune.sh` | Finetuning launcher |
| `/root/groot/scripts/eval_groot_franka.py` | Isaac Sim eval script (L4) |

---

## References
- GR00T GitHub: https://github.com/NVIDIA/Isaac-GR00T
- v1 Checkpoint: https://huggingface.co/tolasing/groot-pick-place
- Fork: https://github.com/tolasing/Isaac-GR00T (branch: `groot_training`)
