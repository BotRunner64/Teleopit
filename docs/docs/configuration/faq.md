---
sidebar_position: 3
---

# Configuration FAQ

## Why does it fail even though I set `policy_path`?

1. Verify the file exists
2. Confirm it's not an old 1402D / TWIST2 ONNX model
3. Confirm the input dimension is `166` with dual inputs (`obs` + `obs_history`)

## Why must I specify `input.bvh_file` explicitly?

`input/bvh.yaml` no longer provides machine-specific default paths. Always specify explicitly:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

## Why doesn't `viewer=true` work?

The legacy `viewer` alias has been removed. Use `viewers` (plural):

```bash
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=sim2sim
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=none
```
