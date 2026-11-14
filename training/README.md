# gstrain — Gaussian Splatting trainer

`gstrain` trains a **3D Gaussian Splatting** scene from a COLMAP reconstruction and
writes a `.ply` file that the Maya plugin in [`../src`](../src) can load.

- Input: a COLMAP project (`sparse/0` + `images/`)
- Output: `point_cloud.ply` (3DGS format, SH up to degree 3), previews, and a
  progress log that other tools (e.g. a Maya panel) can watch
- Pure Python + PyTorch, runs on CUDA (or CPU, very slowly). No Maya dependency.

[`old_training/`](old_training) is the previous trainer, kept **read-only as a
reference**. Its features were ported into `gstrain` (see
[OLD_VS_NEW_TRAINING.md](OLD_VS_NEW_TRAINING.md)). [TRAINING.md](TRAINING.md) is a long
walkthrough written for the old trainer.

---

## 1. Setup

Run once, from this `training/` folder (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[train,dev]"
```

Install a CUDA build of PyTorch if `pip` picks the CPU one
(see https://pytorch.org/get-started/locally/). Check the GPU is found, then run the tests:

```powershell
.venv\Scripts\python.exe -c "from gstrain.torch_utils import describe_device; print(describe_device())"
.venv\Scripts\python.exe -m pytest tests -q                  # all 115 tests, ~5 s
.venv\Scripts\python.exe -m pytest tests -q -m "not slow"    # skip the end-to-end ones, ~2 s
.venv\Scripts\python.exe -m pytest tests\test_cli.py -q      # one file
```

The tests use only synthetic data (no MeSP needed). `tests/conftest.py` builds a tiny
COLMAP project (8 photos of 32×24 px, one PINHOLE and one SIMPLE_RADIAL camera) that the
command-line tests train, clean, verify and resume on. The tests marked `slow` are the
end-to-end runs: `test_cli.py` and the opacity-reset collapse regression.

---

## 2. Running a training

From `training/`:

```powershell
# MeSP with the defaults: 7000 iterations at 480 px, output in MeSP\runs\latest
.venv\Scripts\python.exe main.py

# any scene / any option
.venv\Scripts\python.exe -m gstrain.cli train -s C:\path\to\project -m C:\path\to\output --iterations 7000

# quick check that everything works (~15 s)
.venv\Scripts\python.exe main.py --iterations 60 --max-edge 120 --max-views 12 -m C:\temp\smoke

# what is in a COLMAP project? / check a written .ply
.venv\Scripts\python.exe -m gstrain.cli stats -s C:\Users\Geri\Documents\Projects\CG\MeSP
.venv\Scripts\python.exe main.py verify C:\Users\Geri\Documents\Projects\CG\MeSP\runs\latest\point_cloud.ply
```

`main.py` forwards everything to `gstrain.cli`; `-s` / `-m` on the command line override its
MeSP defaults.

### Options (`train`)

| Option | Default | Meaning |
|---|---|---|
| `-s`, `--source` | — | COLMAP project root (or its `sparse/0`) |
| `-m`, `--model` | — | Output folder |
| `--images` | `<project>/images` | Photo folder |
| `--iterations` | 7000 | One view per iteration (3DGS uses 30000) |
| `--max-edge` | 480 | Long image side in pixels |
| `--max-views` | 0 = all | Use N evenly spaced views |
| `--eval-hold` | 8 | Every 8th view is held out for testing (0 = none) |
| `--max-points` | 0 = all | Starting splats taken from COLMAP points |
| `--sh-degree` | 3 | Spherical-harmonic colour degree (0 = plain RGB) |
| `--background` | black | `black`, `white` or `random` (new colour each iteration) |
| `--strategy` | default | `default` = 3DGS clone/split/prune; `mcmc` = 3DGS-MCMC (§5.2) |
| `--max-gaussians` | 2,000,000 | Splat budget for the default strategy |
| `--cap-max` | 300,000 | Splat cap for the MCMC strategy |
| `--max-screen-fraction` | off | Default strategy: after the first opacity reset also prune splats wider than this fraction of the image (e.g. `0.15`). Off by default, see §5.1 |
| `--prune-contribution` | 0 | After training, drop this fraction of least-contributing splats (e.g. `0.3`), then fine-tune |
| `--prune-finetune` | 500 | Fine-tune iterations after pruning |
| `--eval-every` | 500 | Held-out PSNR/SSIM every N iterations |
| `--preview-every` | 500 | Preview image every N iterations (0 = off) |
| `--save-iterations` | 7000 30000 | Extra `.ply` snapshots |
| `--checkpoint-every` / `--resume` | 0 / — | Save `chkpntN.pt`, continue from one |
| `--up-axis` | colmap | `maya-y` rotates the scene to Maya's Y-up |
| `--seed` | 0 | Random seed |

### Output

| File | Content |
|---|---|
| `point_cloud.ply` | Final model (also `point_cloud/iteration_N/point_cloud.ply`) |
| `metrics.json` | Final held-out PSNR / SSIM / L1, splat count, time |
| `progress.jsonl` | One JSON line every `--log-every` iterations: loss, PSNR, splats, LR, SH degree; splat size/opacity statistics every 100; test metrics; densify / prune events |
| `status.json` | `starting` / `running` / `done` / `error` / `cancelled` |
| `previews/iter_N.jpg` | Held-out view: target / render / error ×4 |
| `cfg_args.json` | The options used |

---

### Cleaning a model for Maya

A model trained on photos of a room also contains **the room** and many **junk splats**
(nearly transparent, or almost never seen). From the photo positions you don't notice them;
orbiting the model in Maya they look like a noisy cloud around the subject. `clean` writes a
smaller, cleaner copy. The trained `point_cloud.ply` is never changed.

#### How it works

`clean` loads the trained `.ply`, re-renders the **training photos**, and applies three filters.
A splat is kept only if it passes all the switched-on filters:

1. **Opacity** (`--min-opacity`, default 0.05). Splats more transparent than 5 % are
   removed. They add almost nothing to any image but form the "haze" when seen from the side.
2. **Contribution** (`--keep-contribution`, default 0.99). Every training photo is rendered
   once and each splat's share of the image is added up (Σ T·α over all pixels: how
   visible it is × how much it is not hidden behind others; the score from the
   LightGaussian paper). Splats are sorted from strongest to weakest, and the fewest strong
   ones that together make **99 %** of the images are kept. Splats that are never really
   seen (hidden behind others, outside every photo) have a tiny score and are removed.
3. **Crop** (`--crop-radius`, off by default). `clean` first finds the **subject**: the point
   that all cameras look at, i.e. the point closest to every camera's viewing direction
   (least squares). Then it keeps only splats within
   `crop radius × median camera distance` of that point. With `0.5` on MeSP that is a
   sphere of radius 1.9 around the head: the person stays, the room goes.

Afterwards it writes the kept splats as a new `.ply` (same format, SH colour included), a
report, and a before/after preview: the first held-out photo view, and the same view swung
40° around the subject (roughly what an orbit in Maya shows).

#### Commands

Run from `C:\Users\Geri\Documents\Projects\CG\MayaGaussianSplatting`. Pass the **run folder**:
scene, resolution and axis are read from its `cfg_args.json`.

```powershell
# 1. remove junk only, keep the room (model looks the same from the photo positions)
.\training\.venv\Scripts\python.exe .\training\main.py clean C:\Users\Geri\Documents\Projects\CG\MeSP\runs\C_7k_480_old

# 2. remove junk AND the room: only the subject is left (for Maya)
.\training\.venv\Scripts\python.exe .\training\main.py clean C:\Users\Geri\Documents\Projects\CG\MeSP\runs\C_7k_480_old --crop-radius 0.5

# 3. gentler: keep more fine detail (removes fewer splats)
.\training\.venv\Scripts\python.exe .\training\main.py clean C:\Users\Geri\Documents\Projects\CG\MeSP\runs\C_7k_480_old --min-opacity 0.02 --keep-contribution 0.995 --crop-radius 0.5

# 4. own output file, and rotated for Maya's Y-up
.\training\.venv\Scripts\python.exe .\training\main.py clean C:\Users\Geri\Documents\Projects\CG\MeSP\runs\C_7k_480_old --crop-radius 0.5 --up-axis maya-y -o C:\Users\Geri\Documents\Projects\CG\MeSP\me_clean.ply

# check the result
.\training\.venv\Scripts\python.exe .\training\main.py verify C:\Users\Geri\Documents\Projects\CG\MeSP\runs\C_7k_480_old\point_cloud_clean.ply
```

#### Options

| Option | Default | Meaning |
|---|---|---|
| `input` | — | Training output folder (or its `point_cloud.ply`) |
| `--min-opacity` | 0.05 | Remove splats below this opacity. 0 = off |
| `--keep-contribution` | 0.99 | Keep the strongest splats that make this share of the training images. 1 = off |
| `--crop-radius` | 0 = off | Keep splats within this many camera distances of the subject |
| `-o`, `--output` | `<run>\point_cloud_clean.ply` | Output file |
| `--up-axis` | as trained | `maya-y` writes the file rotated to Maya's Y-up |
| `-s`, `--images`, `--max-edge`, `--eval-hold` | from `cfg_args.json` | Only needed for a `.ply` without its run folder |
| `--no-preview` | preview on | Skip the before/after image |
| `--preview-angle` | 40 | Side angle of the second preview row |

#### Which setting for which problem

| You see in Maya | Change |
|---|---|
| The whole room around the subject | `--crop-radius 0.5` (smaller = tighter, e.g. `0.35`; larger keeps more, e.g. `0.8`) |
| Part of the subject is cut off (shoulders, hair) | larger `--crop-radius` |
| Light haze / fog everywhere | higher `--min-opacity` (e.g. `0.1`) |
| Fine detail got lost (thin hair, edges) | lower `--min-opacity` (`0.02`) and higher `--keep-contribution` (`0.995`) |
| Haze right around the subject remains | cannot be fixed by `clean` (those splats are inside the crop and visible in the photos); see §7 |

#### Output

| File | Content |
|---|---|
| `point_cloud_clean.ply` | The cleaned model: load this in Maya |
| `point_cloud_clean_preview.jpg` | Before / after, front view and 40° to the side |
| `point_cloud_clean_report.json` | Splats before / after, how many each filter removed, crop centre and radius, file sizes, test PSNR before / after |

The terminal shows the same summary. Test PSNR is measured on the held-out photos, which
**contain the room**, so after cropping it drops a lot (28.6 → 6.3 dB on MeSP). That is expected:
it measures the room that was deliberately removed, not the subject's quality.

#### Results on MeSP run C

| Command | Splats | File | Test PSNR |
|---|---|---|---|
| (trained model) | 195,264 | 46.2 MB | 28.56 dB |
| `clean <run>` | 77,871 | 18.4 MB | 27.15 dB |
| `clean <run> --crop-radius 0.5` | 13,324 | 3.2 MB | 6.27 dB (room removed) |

---

## 3. Files

| File | What it does |
|---|---|
| [`colmap.py`](gstrain/colmap.py), [`cameras.py`](gstrain/cameras.py) | Read COLMAP; intrinsics, poses, lens distortion |
| [`undistort.py`](gstrain/undistort.py) | Rectify photos to the pinhole model on load |
| [`dataset.py`](gstrain/dataset.py) | Load / resize / rectify photos, train/test split |
| [`model.py`](gstrain/model.py) | Starting splats from the COLMAP points |
| [`model_torch.py`](gstrain/model_torch.py), [`sh.py`](gstrain/sh.py) | Trainable splats, spherical-harmonic colour |
| [`render/`](gstrain/render) | Differentiable renderer: projection, 2D covariance, tile binning, rasterizer |
| [`loss.py`](gstrain/loss.py) | L1 + D-SSIM loss, PSNR |
| [`train.py`](gstrain/train.py) | Training loop (`TrainConfig`) |
| [`strategy.py`](gstrain/strategy.py), [`densification.py`](gstrain/densification.py) | Default strategy (3DGS density control), contribution pruning |
| [`strategy_mcmc.py`](gstrain/strategy_mcmc.py) | MCMC strategy |
| [`cli.py`](gstrain/cli.py), [`run_log.py`](gstrain/run_log.py), [`checkpoint.py`](gstrain/checkpoint.py) | Command line, progress files, checkpoints |
| [`export_ply.py`](gstrain/export_ply.py), [`preview.py`](gstrain/preview.py) | `.ply` writer/reader/loader/verifier, preview images |
| [`clean.py`](gstrain/clean.py) | `clean` command: opacity / contribution / crop filters, subject focus point |

---

## 4. How training works

### 4.1 Start

1. Read COLMAP. Sort the images by name; every 8th is a **test view** that is never
   trained on and only used to measure quality.
2. Load the photos, downscale to `--max-edge`, and **rectify lens distortion**
   (MeSP's camera has k1 = 0.025, about 10 px at the corners at full size).
3. One splat per COLMAP point: size = mean distance to its 3 nearest neighbours,
   opacity 0.1, colour from the point (as spherical-harmonic band 0).
4. Scene extent = radius of the camera positions × 1.1; position learning rate and
   size thresholds scale with it.

### 4.2 One iteration

```text
take the next view (all views once per epoch, shuffled)
render it  ->  loss = 0.8 * L1 + 0.2 * (1 - SSIM)  (+ MCMC regularizers)
backward, Adam step
strategy: densify / prune / relocate splats when scheduled
every 1000 iterations: one more SH band (colour may depend on view direction)
```

### 4.3 Renderer

1. Project every splat centre (pinhole; the photos were rectified on load).
2. 2D shape `J W Σ Wᵀ Jᵀ` with the Jacobian clamped to 1.3 × the field of view, plus
   **0.3 px² dilation** so no splat is smaller than about a pixel (EWA anti-aliasing).
3. Each splat ends where its alpha drops below 1/255; its tile box is sized from its
   opacity to match.
4. 16×16 tiles, splats sorted front to back; tiles processed in groups of similar
   occupancy. Alpha blending `C = Σ cᵢ αᵢ Tᵢ + T·background`, α ≤ 0.99, stop once a
   tile is opaque. Pixels are sampled at their centres (i + 0.5, the COLMAP convention).

---

## 5. Controlling the splats

### 5.1 Default strategy (3DGS adaptive density control)

Every 100 iterations in the first half of training:

- **clone** small splats and **split** large ones whose screen-space gradient
  (averaged over the views that saw them) is ≥ 2e-4;
- **prune** splats with opacity < 0.005; after the first opacity reset also splats
  larger than 10 % of the scene extent;
- **opacity reset** every 3000 iterations (as in 3DGS), only while densifying, so useless
  splats fade and get pruned. Runs shorter than ~6000 iterations have no reset.

**No screen-size pruning by default** (same as gsplat). A fixed "larger than 20 px on
screen" rule deleted the splats that cover walls and background: on MeSP at 480 px one
round removed 10,935 splats, every later round removed whatever grew back, and held-out
PSNR fell from 20.8 to 6.4 dB for good. It can be switched on, relative to the image size,
with `--max-screen-fraction`. A round that prunes more than 25 % of all splats now prints a
`WARNING` line.

At 30,000 iterations this is exactly the 3DGS schedule; the training log prints the
schedule at the start.

### 5.2 MCMC strategy (`--strategy mcmc`)

From *3D Gaussian Splatting as Markov Chain Monte Carlo* (Kheradmand et al. 2024):

- a **fixed cap** (`--cap-max`); the splat count grows 5 % every 100 iterations until it
  reaches the cap;
- splats whose opacity drops below 0.005 are **moved** onto useful ones instead of being
  deleted, with opacity and size chosen so the image does not change;
- small **L1 penalties on opacity and size** (0.01 each) make useless splats fade and keep
  splats from becoming large blurry blobs;
- **position noise** for nearly transparent splats lets them escape bad spots instead of
  turning into floaters. No opacity resets.

Good when the splat count must stay bounded (8 GB GPU, file size for Maya).

### 5.3 Contribution pruning (`--prune-contribution 0.3`)

From *LightGaussian* (Fan et al. 2024): after training, measure how much each splat shows
up in the training images (Σ T·α over all pixels), delete the lowest fraction, and fine-tune
for `--prune-finetune` iterations. Shrinks the `.ply` with little quality loss.

---

## 6. Results on MeSP

117 photos (102 train / 15 held out), RTX 3060 Ti. PSNR / SSIM are on the held-out
views (higher is better).

| Run | Setting | Test PSNR | Test SSIM | Splats | Time |
|---|---|---|---|---|---|
| Old trainer | 1000 it., 240×135 | 17.4 * | 0.68 * | 23,003 | 157 s |
| B | 1000 it., 240×135 | **25.3** (26.8 *) | 0.86 | 18,895 | 113 s |

\* The old trainer averaged only the first 4 held-out views; B on those same 4 views is
in brackets.

More runs (480 px, 7000 iterations, SH, MCMC, pruning) are listed in
[OLD_VS_NEW_TRAINING.md](OLD_VS_NEW_TRAINING.md#outcome).

---

## 7. Known limits

- Every rendered pixel is plain PyTorch (no custom CUDA kernel), so very high
  resolutions or millions of splats are slow; lower `tile_chunk` / `depth_chunk` in
  `TrainConfig` if memory runs out.
- `--up-axis maya-y` rotates positions and orientations; higher SH bands are not
  rotated (band 0, the base colour, is unaffected).
- Haze directly around the subject (inside the crop) cannot be removed by `clean`; training
  on masked photos (person cut out) or `--strategy mcmc` are the next steps for that.
- Possible next steps: Mip-Splatting's 3D size filter, AbsGS-style densification
  (needs per-pixel gradients from a custom backward pass).
