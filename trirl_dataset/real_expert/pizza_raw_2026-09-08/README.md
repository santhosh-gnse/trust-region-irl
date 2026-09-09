# Franka Pizza -- Kinesthetic Dataset

Hand-guided demonstrations of pizza-sauce spreading, collected on a real
Franka Research 3 (FR3) arm. This bundle has everything needed to interpret
the data or reconstruct the physical setup, without the rest of the
`franka_pizza` ROS 2 package.

## Robot and tool

- Arm: Franka FR3, base frame `fr3_link0`.
- End effector: `custom_pusher_ee` -- a 3D-printed rod (0.02 x 0.02 x 0.125 m
  box) with a 0.015 m sphere tip. Tool frame `fr3_pusher_tcp` sits at the
  sphere center, offset `xyz: 0 0 0.0625` from the flange.
  `config/custom_pusher_ee_inertials.yaml` has the mass (15 g, measured
  placeholder -- see its own TODO) and inertia used for gravity/dynamics
  compensation.
- `config/fr3_joint_limits.yaml`: the arm's real per-joint position limits,
  included so `home_joint_positions` below can be checked against them.

## Calibration (`config/pizza_robot.yaml`)

- `home_position`: `[0.6247, -0.0094, 0.0641]` in `fr3_link0` -- the board's
  physical centre in x/y (measured via a 3-marker OptiTrack rigid body,
  0.01 mm jitter, through the same hand-eye anchor used elsewhere in this
  robot cell), and z = board-touch height (0.0441 m, measured by FK at zero
  spread over 9453 samples) + 2 cm.
- `home_joint_positions`: the 7 joint values that place `fr3_pusher_tcp` at
  `home_position`, tool pointing straight down. Solved via `/compute_ik` and
  confirmed on the real robot (1.3 mm / 0.29 deg from target). Minimum
  joint-limit margin ~67 deg (joint 4); every other joint has 68-142 deg of
  clearance -- see `fr3_joint_limits.yaml` to re-check if this pose is ever
  revised.
- `orientation_hold_quaternion_wxyz`: `[0.0, 0.7071068, 0.7071068, 0.0]` --
  tool-down, a fixed world-frame quaternion (independent of robot base
  position). Same physical tool as `franka_pusht`; reused from its measured
  value.

## What was and wasn't enforced during collection

`z_hold_target`/`orientation_hold_*` in `config/pizza_common.yaml` describe
the DEPLOYMENT-time constraint (tool locked 2 cm above the board, straight
down) -- they were **not** actively enforced during this kinesthetic
collection. An earlier attempt to lock height/orientation with a stiff
Cartesian controller even while hand-guiding caused sustained oscillation the
moment the tool touched the board (a rigid-on-rigid contact instability, not
a tuning bug). Collection instead used plain gravity-compensated guiding
(fully compliant, all axes) -- height/orientation were only as consistent as
the operator's own hand kept them. Enforcing the lock on a deployed policy's
actions is a future `safety_node`/`servo_ik_node`'s job, not built yet.

## Data format

Each episode is one `.npz` file, one demonstrated spreading motion:

| key | shape | dtype | meaning |
|---|---|---|---|
| `states` | (T, 17) | float32 | `ee_pos_rel_home[3]` (m, in `fr3_link0`) + `arm_qpos[7]` (rad) + `arm_qvel[7]` (rad/s) |
| `actions` | (T, 2) | float32 | `clip(ee_velocity_xy / max_linear_speed, -1, 1)` -- horizontal tool velocity, normalized and clipped |
| `next_states` | (T, 17) | float32 | `states` shifted by one step (`next_states[i] == states[i+1]`) |
| `absorbing` | (T,) | float32 | 1.0 on the last transition of the episode, 0.0 otherwise |
| `rewards` | (T,) | float32 | all zeros -- no reward signal defined for this task |
| `raw_actions_mps` | (T, 2) | float32 | **only in episodes recorded after 2026-09-08 evening** -- same tool velocity in m/s, filtered but NOT normalized or clipped; re-derive `actions` at a different `max_linear_speed` from this instead of re-collecting |

`max_linear_speed = 0.10` m/s (`config/pizza_common.yaml`). Control rate
20 Hz. Height/orientation are deliberately not part of the action or a
separate observation dim -- the task holds both roughly fixed (see caveat
above), so there's nothing 2-D policy needs to command there.

### Known data-quality caveats

- **Action clipping is lossy and can distort direction, not just speed.**
  `np.clip` is applied per-axis, so a step where only one axis exceeded
  0.10 m/s saves a *different* direction than was actually moved, not just a
  capped speed. Across the two sessions here, roughly 40% of steps per
  episode hit this in some form. Episodes recorded before the
  `raw_actions_mps` fix (see table above) cannot be corrected after the fact
  -- the true velocity above the cap was never recorded for those.
- One episode (`kinesthetic_20260908_212040/episode_0056.npz`) briefly
  reached z = -22.5 mm relative to home, i.e. ~2.5 mm past the board surface
  (home is +20 mm above the board) -- a light contact, not an error.

## Sessions in this bundle

- `data/kinesthetic_20260908_212040/` -- 80 episodes
- `data/kinesthetic_20260908_221424/` -- 39 episodes

## Visualization

`viz/spread_trace_review.html` -- a self-contained, offline HTML page
plotting two sampled episodes' tool path, height, and commanded velocity.
Open directly in a browser, no server needed.
