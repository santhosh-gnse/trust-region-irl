# BulbScrew slotting demonstrations — 2026-09-06

**126 episodes, full length, exactly as recorded.** Nothing truncated,
nothing modified: byte-for-byte the recorder's output, gathered into one flat
directory. 31419 transitions, 26.2 minutes at 20 Hz.

## The task

The bulb is **slotted**, not screwed: it starts already held in the gripper,
is carried to the socket, dropped into the mouth and released. Nothing is
threaded, so **any final rotation counts** and the bulb rests at a lean.

An episode is included when it reached `d_seat < 20 mm` held for 1 s, and had a
joint-limit margin above 10° throughout (Servo halts inside 5.7°, so a tighter
episode is one a deployed policy could not reproduce).

## Quality

| | |
| --- | --- |
| mocap dropouts | **0.00%** worst case — none |
| final d_seat | 7.2–8.4 mm, sd **0.27 mm** |
| final tilt | 1.3°–6.1° |
| grasp residual | 8.5–13.3 mm median |
| joint margin | 25°–56° |
| length | 149–365 steps (7.5–18.2 s), mean 249 |

The **0.27 mm** spread in final depth is the socket setting the stop,
not the operator. Success rate on genuine attempts was 98%.

## Known bias — read before training

The tool sits a consistent **~8.5 mm below** the computed grasp point while
gripping, with only 1.7 mm of variation. That is a `bulb_neck_offset`
calibration bias, not slip — it affects every episode here equally and is
correctable in post if you re-measure the grasp offset. It is well inside the
bulb frame's own stated accuracy (the axis came from a 6.70 mm circle with
1.62 mm residual), so it is not alarming, just knowable.

## Format

`states (N,25)`, `actions (N,5)`, `next_states (N,25)`, `absorbing (N,)`,
`rewards (N,)` at 20 Hz.

```
observation = bulb_tip_rel_seat[3] + bulb_quat_rel_seat[4]
            + ee_rel_grasp[3] + gripper_width[1] + arm_qpos[7] + arm_qvel[7]
action      = [vx, vy, vz, yaw_rate, grip]  in [-1, 1]
```

**The first ten dims are in the SEAT frame, not `fr3_link0`** — that frame is
rotated **−77.7°** about z for these episodes. Reconstructing a position without
rotating first moves the bulb ~0.4 m to a plausible wrong place. Use

```python
Q_L0_SEAT = [0.778551, -0.000876, -0.003338, -0.627571]   # wxyz
```

or take positions from forward kinematics on `obs[11:18]`. Episodes recorded
before 2026-09-06 carry a *different* rotation — see `docs/FINDINGS.md` B17.

Actions are scaled by `max_linear_speed = 0.10` m/s and `max_yaw_rate = 0.75`
rad/s. Note the linear axes saturate on ~6.5% of steps: the carry phase is
briefly hand-guided faster than 0.10 m/s, so transport is labelled slower than
performed. The slot itself is unaffected.

## Included

| episode | steps | final d_seat | final tilt | grasp resid | joint margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| kinesthetic_20260906_210131_episode_0000.npz | 278 | 7.3 mm | 4.6° | 8.5 mm | 37° |
| kinesthetic_20260906_210131_episode_0001.npz | 305 | 7.6 mm | 4.7° | 10.3 mm | 43° |
| kinesthetic_20260906_210131_episode_0002.npz | 334 | 8.2 mm | 5.1° | 10.8 mm | 56° |
| kinesthetic_20260906_210131_episode_0003.npz | 276 | 7.5 mm | 6.1° | 8.9 mm | 34° |
| kinesthetic_20260906_210131_episode_0004.npz | 326 | 7.6 mm | 4.9° | 8.7 mm | 41° |
| kinesthetic_20260906_210131_episode_0005.npz | 265 | 7.2 mm | 4.8° | 10.1 mm | 34° |
| kinesthetic_20260906_210131_episode_0006.npz | 246 | 7.2 mm | 4.3° | 8.6 mm | 35° |
| kinesthetic_20260906_210131_episode_0007.npz | 266 | 7.5 mm | 4.5° | 8.6 mm | 29° |
| kinesthetic_20260906_210131_episode_0008.npz | 294 | 7.5 mm | 5.4° | 9.8 mm | 29° |
| kinesthetic_20260906_210131_episode_0009.npz | 231 | 7.9 mm | 3.9° | 11.0 mm | 32° |
| kinesthetic_20260906_210131_episode_0010.npz | 277 | 7.6 mm | 5.5° | 9.1 mm | 28° |
| kinesthetic_20260906_210131_episode_0011.npz | 272 | 7.9 mm | 6.0° | 10.2 mm | 33° |
| kinesthetic_20260906_210131_episode_0012.npz | 271 | 7.6 mm | 5.1° | 10.5 mm | 37° |
| kinesthetic_20260906_210131_episode_0013.npz | 299 | 7.6 mm | 5.1° | 9.9 mm | 36° |
| kinesthetic_20260906_210131_episode_0014.npz | 281 | 8.2 mm | 4.0° | 10.6 mm | 36° |
| kinesthetic_20260906_210131_episode_0015.npz | 254 | 7.6 mm | 4.7° | 11.1 mm | 34° |
| kinesthetic_20260906_210131_episode_0016.npz | 245 | 7.4 mm | 4.8° | 11.6 mm | 26° |
| kinesthetic_20260906_210131_episode_0017.npz | 261 | 7.3 mm | 4.5° | 10.1 mm | 32° |
| kinesthetic_20260906_210131_episode_0018.npz | 303 | 8.1 mm | 3.9° | 12.1 mm | 30° |
| kinesthetic_20260906_210131_episode_0019.npz | 323 | 8.1 mm | 4.2° | 11.2 mm | 31° |
| kinesthetic_20260906_210131_episode_0020.npz | 344 | 8.1 mm | 4.0° | 8.9 mm | 33° |
| kinesthetic_20260906_210131_episode_0021.npz | 259 | 7.5 mm | 5.2° | 10.9 mm | 34° |
| kinesthetic_20260906_210131_episode_0022.npz | 275 | 7.5 mm | 5.3° | 9.5 mm | 33° |
| kinesthetic_20260906_210131_episode_0023.npz | 302 | 8.0 mm | 4.6° | 8.5 mm | 45° |
| kinesthetic_20260906_210131_episode_0024.npz | 280 | 8.2 mm | 4.3° | 9.9 mm | 30° |
| kinesthetic_20260906_210131_episode_0025.npz | 329 | 7.6 mm | 6.0° | 11.1 mm | 29° |
| kinesthetic_20260906_210131_episode_0026.npz | 312 | 8.3 mm | 4.2° | 9.3 mm | 32° |
| kinesthetic_20260906_210131_episode_0027.npz | 317 | 7.6 mm | 5.7° | 8.6 mm | 33° |
| kinesthetic_20260906_210131_episode_0028.npz | 280 | 7.4 mm | 5.3° | 10.6 mm | 33° |
| kinesthetic_20260906_210131_episode_0029.npz | 312 | 8.2 mm | 5.0° | 8.9 mm | 33° |
| kinesthetic_20260906_210131_episode_0030.npz | 334 | 8.3 mm | 3.9° | 9.1 mm | 32° |
| kinesthetic_20260906_210131_episode_0031.npz | 249 | 7.6 mm | 5.3° | 10.6 mm | 27° |
| kinesthetic_20260906_210131_episode_0032.npz | 260 | 7.6 mm | 5.3° | 8.9 mm | 31° |
| kinesthetic_20260906_210131_episode_0033.npz | 245 | 7.9 mm | 4.5° | 9.4 mm | 32° |
| kinesthetic_20260906_210131_episode_0034.npz | 217 | 8.4 mm | 3.8° | 9.3 mm | 32° |
| kinesthetic_20260906_210131_episode_0035.npz | 287 | 7.6 mm | 5.8° | 10.0 mm | 33° |
| kinesthetic_20260906_210131_episode_0036.npz | 319 | 8.0 mm | 4.3° | 9.3 mm | 29° |
| kinesthetic_20260906_210131_episode_0037.npz | 189 | 7.9 mm | 4.1° | 10.5 mm | 29° |
| kinesthetic_20260906_210131_episode_0038.npz | 309 | 7.7 mm | 4.6° | 12.3 mm | 36° |
| kinesthetic_20260906_210131_episode_0039.npz | 220 | 7.2 mm | 5.1° | 9.3 mm | 29° |
| kinesthetic_20260906_210131_episode_0040.npz | 262 | 8.1 mm | 4.7° | 10.1 mm | 30° |
| kinesthetic_20260906_210131_episode_0041.npz | 302 | 8.2 mm | 3.7° | 10.4 mm | 29° |
| kinesthetic_20260906_210131_episode_0042.npz | 263 | 7.6 mm | 4.6° | 10.6 mm | 28° |
| kinesthetic_20260906_210131_episode_0043.npz | 207 | 7.4 mm | 5.0° | 9.7 mm | 29° |
| kinesthetic_20260906_210131_episode_0044.npz | 211 | 8.1 mm | 3.9° | 10.8 mm | 27° |
| kinesthetic_20260906_210131_episode_0045.npz | 347 | 7.7 mm | 4.9° | 10.9 mm | 31° |
| kinesthetic_20260906_210131_episode_0046.npz | 232 | 7.4 mm | 4.8° | 10.2 mm | 27° |
| kinesthetic_20260906_210131_episode_0047.npz | 334 | 7.5 mm | 4.9° | 10.3 mm | 31° |
| kinesthetic_20260906_210131_episode_0048.npz | 365 | 7.7 mm | 4.8° | 13.3 mm | 31° |
| kinesthetic_20260906_210131_episode_0049.npz | 170 | 8.1 mm | 4.0° | 9.9 mm | 32° |
| kinesthetic_20260906_210131_episode_0050.npz | 236 | 8.0 mm | 4.0° | 11.2 mm | 29° |
| kinesthetic_20260906_210131_episode_0051.npz | 218 | 7.5 mm | 4.9° | 11.0 mm | 32° |
| kinesthetic_20260906_210131_episode_0052.npz | 218 | 7.5 mm | 4.9° | 10.7 mm | 32° |
| kinesthetic_20260906_210131_episode_0053.npz | 217 | 7.7 mm | 4.9° | 10.6 mm | 29° |
| kinesthetic_20260906_210131_episode_0054.npz | 218 | 8.0 mm | 4.1° | 10.5 mm | 29° |
| kinesthetic_20260906_210131_episode_0055.npz | 149 | 8.2 mm | 3.4° | 11.0 mm | 30° |
| kinesthetic_20260906_210131_episode_0056.npz | 200 | 8.0 mm | 4.1° | 10.7 mm | 28° |
| kinesthetic_20260906_210131_episode_0057.npz | 221 | 7.5 mm | 4.8° | 10.2 mm | 36° |
| kinesthetic_20260906_210131_episode_0058.npz | 228 | 8.0 mm | 3.7° | 11.1 mm | 33° |
| kinesthetic_20260906_210131_episode_0059.npz | 234 | 8.0 mm | 4.3° | 10.4 mm | 30° |
| kinesthetic_20260906_210131_episode_0060.npz | 261 | 7.6 mm | 5.2° | 9.1 mm | 30° |
| kinesthetic_20260906_210131_episode_0061.npz | 281 | 7.7 mm | 5.1° | 9.8 mm | 28° |
| kinesthetic_20260906_210131_episode_0062.npz | 240 | 7.7 mm | 4.0° | 9.7 mm | 32° |
| kinesthetic_20260906_210131_episode_0063.npz | 216 | 7.8 mm | 3.8° | 11.8 mm | 35° |
| kinesthetic_20260906_210131_episode_0064.npz | 256 | 7.9 mm | 4.3° | 8.9 mm | 33° |
| kinesthetic_20260906_210131_episode_0065.npz | 196 | 7.5 mm | 5.2° | 10.0 mm | 31° |
| kinesthetic_20260906_210131_episode_0066.npz | 274 | 7.5 mm | 4.8° | 11.5 mm | 38° |
| kinesthetic_20260906_210131_episode_0067.npz | 218 | 7.7 mm | 4.4° | 9.3 mm | 36° |
| kinesthetic_20260906_210131_episode_0068.npz | 291 | 8.1 mm | 4.0° | 10.3 mm | 37° |
| kinesthetic_20260906_210131_episode_0069.npz | 305 | 8.1 mm | 4.2° | 11.6 mm | 34° |
| kinesthetic_20260906_210131_episode_0070.npz | 255 | 7.9 mm | 4.2° | 10.5 mm | 31° |
| kinesthetic_20260906_210131_episode_0071.npz | 198 | 7.6 mm | 5.0° | 9.5 mm | 30° |
| kinesthetic_20260906_210131_episode_0072.npz | 231 | 7.9 mm | 4.0° | 11.1 mm | 28° |
| kinesthetic_20260906_210131_episode_0073.npz | 185 | 7.5 mm | 5.0° | 10.2 mm | 31° |
| kinesthetic_20260906_210131_episode_0074.npz | 259 | 8.1 mm | 4.0° | 11.1 mm | 28° |
| kinesthetic_20260906_210131_episode_0075.npz | 245 | 8.1 mm | 3.5° | 9.7 mm | 27° |
| kinesthetic_20260906_210131_episode_0076.npz | 282 | 8.2 mm | 4.2° | 13.0 mm | 30° |
| kinesthetic_20260906_210131_episode_0077.npz | 280 | 7.6 mm | 5.0° | 10.9 mm | 31° |
| kinesthetic_20260906_210131_episode_0078.npz | 249 | 7.7 mm | 5.1° | 10.5 mm | 30° |
| kinesthetic_20260906_210131_episode_0079.npz | 187 | 8.1 mm | 3.8° | 9.3 mm | 33° |
| kinesthetic_20260906_210131_episode_0080.npz | 240 | 8.0 mm | 4.4° | 9.8 mm | 27° |
| kinesthetic_20260906_210131_episode_0081.npz | 238 | 8.2 mm | 4.2° | 11.1 mm | 34° |
| kinesthetic_20260906_210131_episode_0082.npz | 223 | 7.9 mm | 4.5° | 9.7 mm | 33° |
| kinesthetic_20260906_210131_episode_0083.npz | 166 | 7.7 mm | 5.1° | 9.8 mm | 32° |
| kinesthetic_20260906_210131_episode_0084.npz | 237 | 8.0 mm | 4.0° | 9.9 mm | 33° |
| kinesthetic_20260906_210131_episode_0085.npz | 256 | 8.1 mm | 3.8° | 9.8 mm | 33° |
| kinesthetic_20260906_210131_episode_0086.npz | 284 | 8.0 mm | 4.6° | 8.9 mm | 28° |
| kinesthetic_20260906_210131_episode_0087.npz | 223 | 7.7 mm | 5.3° | 8.6 mm | 34° |
| kinesthetic_20260906_210131_episode_0088.npz | 224 | 8.0 mm | 4.3° | 9.8 mm | 32° |
| kinesthetic_20260906_210131_episode_0089.npz | 226 | 7.7 mm | 4.9° | 10.0 mm | 26° |
| kinesthetic_20260906_210131_episode_0090.npz | 202 | 7.5 mm | 4.7° | 10.1 mm | 32° |
| kinesthetic_20260906_210131_episode_0091.npz | 210 | 7.9 mm | 4.5° | 11.2 mm | 31° |
| kinesthetic_20260906_210131_episode_0092.npz | 258 | 8.1 mm | 4.3° | 9.4 mm | 32° |
| kinesthetic_20260906_210131_episode_0093.npz | 216 | 8.1 mm | 4.2° | 10.2 mm | 32° |
| kinesthetic_20260906_210131_episode_0094.npz | 190 | 8.0 mm | 4.1° | 11.3 mm | 34° |
| kinesthetic_20260906_210131_episode_0095.npz | 192 | 8.0 mm | 4.3° | 10.3 mm | 30° |
| kinesthetic_20260906_210131_episode_0096.npz | 188 | 8.0 mm | 4.1° | 10.7 mm | 27° |
| kinesthetic_20260906_210131_episode_0097.npz | 222 | 7.8 mm | 4.9° | 10.2 mm | 36° |
| kinesthetic_20260906_210131_episode_0098.npz | 208 | 8.0 mm | 3.9° | 9.3 mm | 29° |
| kinesthetic_20260906_210131_episode_0099.npz | 161 | 7.5 mm | 5.0° | 10.3 mm | 37° |
| kinesthetic_20260906_210131_episode_0100.npz | 223 | 7.7 mm | 4.7° | 9.6 mm | 34° |
| kinesthetic_20260906_210131_episode_0101.npz | 231 | 8.0 mm | 4.5° | 9.7 mm | 32° |
| kinesthetic_20260906_210131_episode_0102.npz | 272 | 8.0 mm | 4.2° | 10.7 mm | 34° |
| kinesthetic_20260906_210131_episode_0103.npz | 211 | 7.6 mm | 5.1° | 10.9 mm | 36° |
| kinesthetic_20260906_210131_episode_0104.npz | 248 | 7.4 mm | 4.9° | 11.3 mm | 34° |
| kinesthetic_20260906_210131_episode_0105.npz | 240 | 8.0 mm | 4.3° | 9.6 mm | 35° |
| kinesthetic_20260906_210131_episode_0106.npz | 211 | 7.9 mm | 4.2° | 9.9 mm | 35° |
| kinesthetic_20260906_210131_episode_0107.npz | 232 | 7.5 mm | 4.7° | 9.7 mm | 37° |
| kinesthetic_20260906_210131_episode_0108.npz | 228 | 7.6 mm | 4.4° | 10.0 mm | 29° |
| kinesthetic_20260906_210131_episode_0109.npz | 219 | 8.0 mm | 4.2° | 9.4 mm | 29° |
| kinesthetic_20260906_210131_episode_0110.npz | 319 | 8.0 mm | 4.3° | 9.4 mm | 31° |
| kinesthetic_20260906_210131_episode_0111.npz | 300 | 8.0 mm | 4.2° | 9.5 mm | 25° |
| kinesthetic_20260906_210131_episode_0112.npz | 200 | 8.1 mm | 3.8° | 9.6 mm | 36° |
| kinesthetic_20260906_210131_episode_0113.npz | 193 | 7.6 mm | 5.3° | 9.9 mm | 36° |
| kinesthetic_20260906_210131_episode_0114.npz | 216 | 8.0 mm | 4.4° | 10.6 mm | 34° |
| kinesthetic_20260906_210131_episode_0116.npz | 204 | 7.9 mm | 4.2° | 10.0 mm | 27° |
| kinesthetic_20260906_210131_episode_0117.npz | 219 | 7.6 mm | 4.5° | 11.6 mm | 37° |
| kinesthetic_20260906_210131_episode_0118.npz | 279 | 8.0 mm | 1.3° | 10.6 mm | 28° |
| kinesthetic_20260906_210131_episode_0119.npz | 178 | 7.8 mm | 1.6° | 11.1 mm | 31° |
| kinesthetic_20260906_210131_episode_0120.npz | 187 | 8.0 mm | 4.2° | 8.5 mm | 25° |
| kinesthetic_20260906_210131_episode_0121.npz | 233 | 8.0 mm | 4.4° | 10.1 mm | 36° |
| kinesthetic_20260906_210131_episode_0122.npz | 282 | 8.0 mm | 4.5° | 9.8 mm | 33° |
| kinesthetic_20260906_210131_episode_0123.npz | 207 | 8.1 mm | 3.8° | 10.5 mm | 31° |
| kinesthetic_20260906_210131_episode_0124.npz | 275 | 8.0 mm | 4.1° | 9.3 mm | 36° |
| kinesthetic_20260906_210131_episode_0125.npz | 300 | 8.1 mm | 3.8° | 10.5 mm | 37° |
| kinesthetic_20260906_210131_episode_0126.npz | 161 | 8.0 mm | 4.2° | 10.0 mm | 36° |

## Excluded (13)

| episode | steps | why excluded |
| --- | ---: | --- |
| kinesthetic_20260906_201003_episode_0000.npz | 1637 | never reached d_seat < 20 mm |
| kinesthetic_20260906_202221_episode_0000.npz | 17 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0001.npz | 8 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0002.npz | 1 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0003.npz | 5 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0004.npz | 4 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0005.npz | 1 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0006.npz | 32 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0007.npz | 1 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0008.npz | 13 | under 2 s -- accidental press |
| kinesthetic_20260906_202221_episode_0009.npz | 17 | under 2 s -- accidental press |
| kinesthetic_20260906_204630_episode_0000.npz | 852 | joint-limit margin 1.4 deg -- Servo halts inside 5.7 |
| kinesthetic_20260906_210131_episode_0115.npz | 79 | never reached d_seat < 20 mm |
