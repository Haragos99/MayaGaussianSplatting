# gstrain — Gaussian Splatting trainer

`gstrain` trains a **3D Gaussian Splatting** scene from a COLMAP reconstruction and
writes a `.ply` file that the Maya plugin in [`../src`](../src) can load.

- Input: COLMAP `sparse/0` (cameras, poses, sparse points) + the `images/` folder
- Output: `point_cloud.ply` (3DGS format) + preview images during training
- Pure Python + PyTorch, runs on CUDA (or CPU, very slowly). No Maya dependency.

For a long, stage-by-stage explanation see [TRAINING.md](TRAINING.md). Parts of it
describe older file names (for example `params.py`, `torch_raster.py`), so if it
disagrees with this README, trust the code and this file.

---

## 1. Setup

Run once, from this `training/` folder (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[train,dev]"
```

Install a CUDA build of PyTorch if `pip` picks the CPU one
(see https://pytorch.org/get-started/locally/). Check the GPU is found:

```powershell
.venv\Scripts\python.exe -c "from gstrain.torch_utils import describe_device; print(describe_device())"
```

Run the tests:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

---

## 2. Running a training

All commands below are run from this `training/` folder.

### 2.1 With `main.py`

```powershell
.venv\Scripts\python.exe main.py --iterations 1000 --max-edge 256 --max-gaussians 20000
```

**Important:** `main()` still runs some old debug code *before* training starts:
a CUDA speed test and a test render that opens a **matplotlib window**. Training
only starts after you **close that window**. This also means `--help` only shows
after the window is closed.

| Option | Default | Meaning |
|---|---|---|
| `colmap_dir` (positional) | `C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0` | COLMAP model folder |
| `--images-dir` | `<project>/images` | Folder with the photos |
| `--iterations` | 1000 | Training iterations |
| `--max-views` | all | Use only the first N views (faster, at least 2) |
| `--max-edge` | 256 | Images are downscaled so the long side is at most this many pixels |
| `--max-gaussians` | 5000 | Max starting splats; extra COLMAP points are dropped evenly. Set it above the point count (MeSP has 10,895) to keep all points |
| `--output-dir` | `out/previews` | Preview folder |
| `--preview-every` | 100 | Save a preview every N iterations (uses the first view) |
| `--ply-output` | `<output-dir>/../point_cloud.ply` | Where the `.ply` is written |

### 2.2 Without the debug code

This calls the training function directly, with no matplotlib window:

```powershell
.venv\Scripts\python.exe -c "from pathlib import Path; from main import run_training_test; run_training_test(Path(r'C:\Users\Geri\Documents\Projects\CG\MeSP\sparse\0'), iterations=1000, max_edge=256, max_gaussians=20000)"
```

`run_training_test` takes the same options as the table above, written in
`snake_case` (`max_views=12`, `preview_every=50`, ...). Its own `iterations`
default is 3, so always pass `iterations`.

The `gstrain` console command in `pyproject.toml` points to `gstrain.cli`, which
does not exist yet, so it does not work.

### 2.3 What you get

| Output | Where (relative to where you ran the command) |
|---|---|
| Preview every `preview_every` iterations (target / render / error ×4) | `out/previews/step_000100.jpg`, ... |
| Trained splats | `out/point_cloud.ply` |

The terminal shows the densification schedule, one line per densify round, the
loss and the splat count (see §5 "Reading the log").

### 2.4 How long it takes

Measured on an RTX 3060 Ti with the MeSP scene (117 views):

| Setting | Time per iteration | 1000 iterations |
|---|---|---|
| all 117 views, 256 px | ~6 s | ~1 h 40 min |
| 12 views (`--max-views 12`), 256 px | ~0.6 s (estimated: time scales with the view count) | ~10 min |

Every iteration renders **all** training views (see §4.3), so the time grows with
the number of views. For a quick check that everything works, use
`--iterations 5`. For quick experiments, `--max-views 12` is much faster, but the
model only learns those 12 photos.

### 2.5 Choosing the number of iterations

The densification schedule follows the run length (§5):

| `--iterations` | What happens |
|---|---|
| ≤ 200 | **No densification.** Only the starting splats are trained, so the result stays blurry. Good only to check that the pipeline runs |
| 500 | 2 densify rounds (100, 200) |
| 1000 (default) | 4 densify rounds (100 … 400) |
| ≥ 1,600 | Densify rounds, opacity resets and size pruning (full 3DGS behaviour) |

Low resolution (`--max-edge 128`) also limits how sharp the result can be.

---

## 3. Files

| File | What it does |
|---|---|
| [`gstrain/colmap.py`](gstrain/colmap.py) | Reads COLMAP `cameras/images/points3D` (`.bin` or `.txt`) |
| [`gstrain/cameras.py`](gstrain/cameras.py), [`geometry/camera.py`](gstrain/geometry/camera.py) | Camera intrinsics/poses, downscaling to `max_edge` |
| [`gstrain/dataset.py`](gstrain/dataset.py) | Loads and resizes the target photos (`TrainView` = camera + image) |
| [`gstrain/model.py`](gstrain/model.py) | Builds the starting splats from the COLMAP points (NumPy) |
| [`gstrain/model_torch.py`](gstrain/model_torch.py) | The trainable splats (`TorchGaussianModel`) |
| [`gstrain/render/`](gstrain/render) | Differentiable renderer: projection → 2D covariance → tile binning → rasterizer |
| [`gstrain/loss.py`](gstrain/loss.py) | L1 + D-SSIM loss |
| [`gstrain/train.py`](gstrain/train.py) | Training loop, optimizer, learning rates (`TrainConfig`) |
| [`gstrain/densification.py`](gstrain/densification.py) | Clone / split / prune / opacity reset (`DensificationConfig`) |
| [`gstrain/preview.py`](gstrain/preview.py) | Preview images (target / render / error) |
| [`gstrain/export_ply.py`](gstrain/export_ply.py) | Writes the 3DGS `.ply` |
| [`main.py`](main.py) | Command line + `run_training_test` (the entry point above) + debug code |
| [`tests/`](tests) | `pytest` tests for densification and the rasterizer |

---

## 4. How training works

### 4.1 Start

1. **Load COLMAP.** Camera poses, intrinsics and the sparse point cloud.
2. **Create one splat per COLMAP point** ([`model.py`](gstrain/model.py)):
   - position = the point
   - colour = the point's RGB
   - size = half the distance to its nearest neighbour (same in x, y, z)
   - rotation = identity, opacity = 0.5
3. **Load the photos**, downscaled to `max_edge`.
4. **Scene extent** = radius of the camera positions × 1.1. Learning rate for
   position and the size thresholds for densification are multiplied by it, so
   the trainer works for any scene scale.

### 4.2 What a splat stores

Each splat has 5 trainable parameters. Some are stored in a raw form so that the
optimizer can never produce an invalid value:

| Parameter | Stored as | Real value |
|---|---|---|
| position `xyz` | as is | 3D point |
| size `scale` | `log(scale)` | `exp(...)` → always positive |
| rotation | quaternion `[w, x, y, z]` | normalised when used |
| opacity | `logit(opacity)` | `sigmoid(...)` → between 0 and 1 |
| colour | `logit(rgb)` | `sigmoid(...)` → between 0 and 1 (no view-dependent colour yet) |

### 4.3 One iteration

```text
for every training view:
    render the splats from this camera
    loss = 0.8 * L1(render, photo) + 0.2 * (1 - SSIM(render, photo))
    backward()                 # gradients for all splats
    remember screen-space gradients (for densification)
average the gradients over the views
Adam step                      # update all splats once
maybe densify / prune / reset opacity
```

Note: every iteration uses **all** views (full batch). The original 3DGS uses
one random view per iteration and many more iterations. Changing this is in the
plan (§7).

### 4.4 Rendering (the differentiable rasterizer)

[`render/renderer_torch.py`](gstrain/render/renderer_torch.py):

1. **Project** each splat centre to the image (pinhole camera).
2. **2D shape:** the 3D covariance `R S Sᵀ Rᵀ` is turned into a 2D ellipse in pixels
   with the projection's Jacobian (`J W Σ Wᵀ Jᵀ`).
3. **Footprint size:** each ellipse is cut off where its alpha drops below
   **1/255**. A fully opaque splat reaches about 3.33σ; faint splats are smaller.
4. **Tile binning:** the image is split into 16×16 pixel tiles. Every splat is
   added to the tiles its footprint touches, and each tile's list is sorted front
   to back by depth.
5. **Alpha blending** per pixel, front to back:
   `colour = Σ cᵢ αᵢ Tᵢ`, where `Tᵢ = Π(1 − αⱼ)` for the splats in front.
   Splats are processed in slices of 256 (`depth_chunk`); once a whole group of
   tiles is opaque (T < 1e-4) the rest is skipped. There is **no cap** on splats
   per tile (`max_per_tile=None`) — a cap drops visible splats and causes square
   seams.

Everything is written with PyTorch operations, so `backward()` gives gradients
for every splat parameter automatically. No custom CUDA kernel.

### 4.5 Learning rates

| Parameter | Learning rate |
|---|---|
| position | 1.6e-4 × extent → 1.6e-6 × extent (exponential decay over the run) |
| scale | 5e-3 |
| rotation | 1e-3 |
| opacity | 5e-2 |
| colour | 2.5e-3 |

These are the 3DGS values (`TrainConfig` in [`train.py`](gstrain/train.py)).

---

## 5. Densification (adding and removing splats)

The starting splats are too few and too large for fine detail. Every `interval`
iterations [`densification.py`](gstrain/densification.py) changes the set:

1. **Statistics.** During each render the gradient of every splat's 2D screen
   position is measured (converted to NDC units, like 3DGS) and averaged over the
   views where the splat was visible. A large average gradient means "this area is
   not reproduced well, the splat wants to move".
2. **Clone** — gradient ≥ `grad_threshold` **and** small
   (largest size ≤ `percent_dense × extent`): an exact copy is added. Training then
   pulls the two apart. This fills empty areas.
3. **Split** — gradient ≥ `grad_threshold` **and** large: the splat is replaced by
   `split_into` (2) children. Their positions are sampled from the parent's own
   Gaussian shape, their size is parent / 1.6, colour, opacity and rotation are
   copied. This adds detail where one big splat covers too much.
4. **Prune** — remove splats that are almost transparent (opacity ≤ `min_opacity`).
   After the first opacity reset also remove splats that are too big on screen
   (> `max_screen_radius` px) or in the world (> `max_scale_fraction × extent`).
   If the schedule has no opacity reset (short runs), size pruning never happens.
   Pruning big splats without a reset first deletes the ones covering walls and
   background and leaves black holes.
5. **Opacity reset** — every `opacity_reset_interval` iterations all opacities are
   capped to 0.01. Splats that are really needed become opaque again quickly;
   useless ones stay transparent and are pruned. Resets only happen while
   densification is still running.

The Adam optimizer's state is kept for surviving splats and starts at zero for
new ones.

### Schedule

`DensificationConfig.for_iterations(n)` builds a 3DGS-style schedule for your run
length. Reference 3DGS (30,000 steps) densifies every 100 steps from 500 to
15,000 and resets opacity every 3,000 steps.

The interval is always **100 optimizer steps**, whatever the run length. A clone
starts exactly on top of its parent, and split children start un-tuned. They need
many Adam steps to move apart and settle before their gradient can tell whether
they need another split. If rounds come faster than that, the same areas are split
again and again before the previous split has done anything, so the splat count
grows without the image getting better.

| | Rule | 500 it. | 1,000 it. | 7,000 it. | 30,000 it. (= 3DGS) |
|---|---|---|---|---|---|
| interval | always 100 (as in 3DGS) | 100 | 100 | 100 | 100 |
| start (exclusive) | n / 60 | 8 | 16 | 116 | 500 |
| end (exclusive) | n / 2 | 250 | 500 | 3,500 | 15,000 |
| rounds | | 2 (100, 200) | 4 (100 … 400) | 33 (200 … 3,400) | 144 (600 … 14,900) |
| opacity reset | every max(n / 10, 800), only before end | off | off | 800, 1600, 2400, 3200 | 3000 … 12,000 |

An opacity reset needs about 100 steps to recover (from 0.01 back to 0.5), so
resets are at least 8 rounds apart. Below about 1,600 iterations there is no reset (and so no size pruning, see step 4 above). Below 200 iterations there is no densification at all.

At the start of training the schedule is printed, for example:

```text
Densification: 2 rounds, every 100 iterations from 100 to 200; opacity reset at: off
```

After the end, the number of splats stays fixed and training only refines them.

A split also happens only where it is needed. A splat is split only when
**all** of these are true:

1. its screen-space gradient, averaged over every view that saw it since the last
   round, is ≥ `grad_threshold` (2e-4);
2. it is large (biggest axis > `percent_dense × extent`); small splats are cloned
   instead;
3. it was not just marked for pruning;
4. the splat budget (`max_gaussians`) is not full (if it is, the highest-gradient
   ones win).

### Reading the log

Example line (numbers are illustrative):

```text
Densification at iteration 100: Gaussians 10895 -> 12410
(candidates clone=1300, split=420; added clone=1300, split=420;
 pruned=205 [opacity=205, screen=0, world=0])
```

- `clone` / `split` candidates should be **above zero** during the first half.
  If they are always 0, `grad_threshold` is too high for your scene.
- If `pruned` removes most splats every time, check opacity and the size limits.
- The splat count is also shown in the progress bar (`splats=`).

### Reading the previews

Each preview has three panels: **target** photo, **render**, **error ×4**
(difference, made 4× brighter). Bright areas in the error panel are where the
model is still wrong — usually edges and fine texture early on.

---

## 6. Configuration reference

To change values, pass your own config to `train()` (in Python):

```python
from gstrain.train import TrainConfig, train
from gstrain.densification import DensificationConfig

config = TrainConfig(
    iterations=500,
    densification=DensificationConfig.for_iterations(500, grad_threshold=1e-4),
)
losses = train(model, views, config=config)
```

`TrainConfig` ([`train.py`](gstrain/train.py)):

| Field | Default | Meaning |
|---|---|---|
| `iterations` | 7000 | Number of optimizer steps |
| `lambda_dssim` | 0.2 | Weight of the SSIM part of the loss |
| `tile_size` | 16 | Tile size in pixels |
| `tile_chunk` | 64 | Tiles rendered at once (lower = less GPU memory) |
| `max_per_tile` | None | Optional cap on splats per tile (causes seams) |
| `depth_chunk` | 256 | Splats per tile blended at once (lower = less GPU memory) |
| `densification` | `DensificationConfig()` | See below |

`DensificationConfig` ([`densification.py`](gstrain/densification.py)):

| Field | Default | Meaning |
|---|---|---|
| `grad_threshold` | 2e-4 | Average screen gradient needed to clone/split |
| `percent_dense` | 0.01 | Clone if size ≤ this × extent, otherwise split |
| `split_into` | 2 | Children per split |
| `min_opacity` | 0.005 | Prune below this opacity |
| `max_screen_radius` | 20 | Prune if larger on screen (pixels) |
| `max_scale_fraction` | 0.1 | Prune if larger than this × extent |
| `start_iteration` / `end_iteration` / `interval` | 50 / 15000 / 100 | When to densify (use `for_iterations` instead) |
| `opacity_reset_interval` / `opacity_reset_value` | 3000 / 0.01 | Opacity reset |
| `max_gaussians` | 2,000,000 | Upper limit; highest-gradient candidates win |

### GPU memory

Memory per view, forward + backward, MeSP scene, 10,895 splats:

| `max_edge` | Peak memory |
|---|---|
| 256 | ~1.2 GB |
| 512 | ~2.3 GB |

The views are rendered one after another, so this does not grow with the number
of views. If you run out of memory, lower `tile_chunk` or `depth_chunk`.

---

## 7. Known problems and next steps

Already fixed:

- Tile-shaped square blocks in the render (per-tile cap of 256 removed).
- Hard rectangular splat edges (3σ box clip replaced by the 1/255 alpha cutoff).
- Split children used the same random noise in every round.
- Opacity reset could happen on the last iteration and make the result transparent.
- The densification schedule did not fit the run length (it never stopped on
  short runs). It is now built from the run length, with a fixed 100-step interval.
- Size pruning ran on runs without an opacity reset. It deleted the big splats
  covering the background and left black holes. On MeSP (100 iterations, 128 px,
  all views) the loss jumped from 0.19 to 0.46 in one round and finished at 0.37.
  After the fix it finished at 0.176, compared with 0.188 for the original code.
- When the splat budget was full, low-index instead of high-gradient candidates were kept.

How to check a change: compare the final `Loss:` line of two runs with the same
settings, and look at the error panel of the last preview.

Still open (in priority order):

1. **Starting splats:** about 8% of the MeSP points are duplicates and start
   with size 0, so they are invisible and never train. Use the mean distance of
   the 3 nearest neighbours with a minimum size, start opacity at 0.1 (3DGS value),
   and keep all COLMAP points by default.
2. **One random view per iteration** instead of all views. Much faster, and it
   matches the 3DGS learning-rate and densification schedules.
3. **Higher resolution** (`max_edge` 512–800) for facial detail; scale
   `max_screen_radius` with the resolution.
4. **Held-out test views** with PSNR in the log, to measure each change.
5. **Spherical harmonics** for view-dependent colour.
6. **Absolute-gradient densification (AbsGS)**, so large blurry splats over fine
   texture also get split.
7. Clean up `main.py` (remove the debug code that opens a window before
   training, add a real `gstrain` command).
