# Droid X2 Mimic Motion Trakcing Lab Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)


## Overview

Based on BeyondMimic , is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.


## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking/
```
## Lafan1 Motion Retargeting
Refer to this repository 
[![License]](https://github.com/wenconggan/Retarget_Motion_Lafan1.git)

  

## Motion Tracking


```bash

python scripts/csv_to_npz_x2.py \
  --input_file /xxx/xxx.csv \
  --input_fps 30 \
  --output_name {motion_name} \
  --save_to xxx.npz \
  --no_wandb 
  --headless
'''

'''
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

This will automatically upload the processed motion file to the WandB registry with output name {motion_name}.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:
；
```bash

# 重放NPZ文件的命令示例：
python scripts/replay_npz_x2.py \
  --motion_file /path/xxx.npz \
  --headless                                         
'''

python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

- Train policy by the following command:

```bash

 python scripts/rsl_rl/train.py --task=x2_mimic --headless --motion_file /home/xxx/whole_body_tracking/motion/chars.npz

python scripts/rsl_rl/train.py --task=x2_mimic --headless --motion_file /path/xxx.npz
```


### Policy Evaluation

- Play the trained policy by the following command:

```bash

python scripts/rsl_rl/play.py \
    --task x2_mimic \
    --num_envs=2 \
    --motion_file /path/xxx.npz \
    --resume_path /path/xxx.pt

python scripts/rsl_rl/play.py \
    --task x2_mimic \
    --num_envs=2 \
    --motion_file /home/wenconggan/whole_body_tracking/motion/chars.npz \
    --resume_path /home/wenconggan/whole_body_tracking/logs/rsl_rl/x2_flat/2025-09-01_11-17-33/model_4000.pt


```

- Sim2Sim Mujoco :

```bash
python scripts/rsl_rl/sim2sim.py  --policy_path /path/xxx.onnx --motion_file /path/xxx.npz


python scripts/rsl_rl/sim2sim.py  --policy_path /home/wenconggan/whole_body_tracking/logs/rsl_rl/x2_flat/2025-09-02_13-12-43/exported/policy.onnx --motion_file /home/wenconggan/whole_body_tracking/motion/chars.npz

```

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies,evaluating trained policies and Sim2sim in Mujuco.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.
