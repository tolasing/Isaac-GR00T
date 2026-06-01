# GR00T Training Context — L40 Machine Summary
> Share this with Claude on the L4 machine before starting Isaac Sim eval.

---

## What Was Done on This Machine (L40, 48GB VRAM)

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
- State: `[wrist_x, wrist_y, wrist_z, gripper]` — 4D, world coords (metres)
- Action: `[d_wrist_x, d_wrist_y, d_wrist_z, d_gripper]` — 4D relative delta
- Video: `observation.images.top` (720×1280, overhead camera)

### 3. Modality Config
Two files created for GR00T to understand the data format:

- `modality_human_hand.py` — Python config registered with GR00T (used during training)
- `modality_human_hand.json` — JSON version (reference)
- `lerobot_dataset/meta/modality.json` — dataset-side modality mapping

**Key config decisions:**
- Embodiment tag: `new_embodiment` (human hand is not a pre-registered embodiment)
- `single_arm` action type: `NON_EEF` (only 3D position, not full 6DOF pose)
- Action representation: `RELATIVE` (delta actions)
- 16-step prediction horizon

### 4. Finetuning
**Command used:**
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
- Base model: `nvidia/GR00T-N1.7-3B` (~3B params, downloaded from HuggingFace)
- Trainable parameters: 1.62B (51.54%) — projector + diffusion head only
- Stopped at step ~3000 (to avoid overfitting on 28 episodes)
- Final loss: ~0.058–0.069
- Checkpoints saved: `checkpoint-1000`, `checkpoint-2000`, `checkpoint-3000`

**HuggingFace access required (gated models):**
- `nvidia/GR00T-N1.7-3B` — accept at huggingface.co/nvidia/GR00T-N1.7-3B
- `nvidia/Cosmos-Reason2-2B` — accept at huggingface.co/nvidia/Cosmos-Reason2-2B

### 5. Checkpoint
Best checkpoint (`checkpoint-2000`) uploaded to HuggingFace:

```
https://huggingface.co/tolasing/groot-pick-place
```

Download on L4:
```bash
huggingface-cli download tolasing/groot-pick-place \
  --local-dir /workspace/checkpoints/groot_pick_place/checkpoint-2000
```

---

## What To Do on the L4 Machine (Isaac Sim Eval)

### Task
Pick-and-place: cube A → B → A, evaluated on Franka Panda in Isaac Sim.

### Action Space Mapping
Training used 4D human hand deltas `[dx, dy, dz, d_gripper]`.
The inference server must map GR00T output → Franka Panda joint commands.
This is configured via `--embodiment franka` in the inference script.

### Eval Steps

**Terminal 1 — Start policy server:**
```bash
python scripts/inference_service.py \
  --model-path /workspace/checkpoints/groot_pick_place/checkpoint-2000 \
  --embodiment franka \
  --port 8000
```

**Terminal 2 — Run Isaac Sim eval:**
```bash
cd /workspace/isaaclab
python source/standalone/gr00t/eval_gr00t.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --policy-server http://localhost:8000 \
  --num-episodes 50
```

---

## Key Files in This Repo (branch: groot_training)

| File | Purpose |
|------|---------|
| `modality_human_hand.py` | GR00T modality config for training |
| `modality_human_hand.json` | JSON reference of modality mapping |
| `lerobot_dataset/meta/modality.json` | Dataset-side modality config |
| `lerobot_dataset/meta/info.json` | Dataset metadata (fixed `episode_chunk` key) |
| `examples/finetune.sh` | Finetuning launcher script |

---

## References
- GR00T GitHub: https://github.com/NVIDIA/Isaac-GR00T
- Checkpoint: https://huggingface.co/tolasing/groot-pick-place
- Fork: https://github.com/tolasing/Isaac-GR00T (branch: groot_training)
