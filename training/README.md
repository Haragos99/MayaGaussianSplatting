# gstrain — Gaussian Splatting trainer

Standalone trainer: COLMAP reconstruction in, 3DGS `.ply` out. **No Maya dependency** —
it runs in a normal Python environment and is fully testable from a terminal. The Maya
side (a UI panel, later) will only launch this as a subprocess and read the files it writes.

## Environment

Maya 2026.2 ships Python 3.11, so the trainer targets 3.11 (3.10 also works). Create a
separate virtual environment — do **not** install anything into Maya's Python.

```powershell
py -3.11 -m venv C:\gs-venv
C:\gs-venv\Scripts\python.exe -m pip install --upgrade pip
C:\gs-venv\Scripts\pip.exe install -r requirements.txt
```

That pulls the CPU build of torch, which is enough for the tests. For GPU training install
the CUDA wheel instead — note `--force-reinstall`, otherwise pip considers the CPU build to
already satisfy the requirement:

```powershell
C:\gs-venv\Scripts\pip.exe install --force-reinstall --no-deps torch `
    --index-url https://download.pytorch.org/whl/cu128
C:\gs-venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

No CUDA Toolkit is required — the wheel ships its own runtime. You only need the toolkit if
you later build custom kernels.

Then install the package in editable mode from this folder:

```powershell
C:\gs-venv\Scripts\pip.exe install -e .
```

## Usage

```powershell
# what does the reconstruction contain?
python -m gstrain.cli stats -s C:\data\myscene

# train (writes previews, progress.jsonl and .ply checkpoints into the output dir)
python -m gstrain.cli train -s C:\data\myscene -m C:\data\myscene\out -r 2 --iterations 30000

# export the initial Gaussians (sparse cloud) as a .ply the Maya plugin can load
python -m gstrain.cli export-init -s C:\data\myscene -m C:\data\myscene\out

# decode any .ply the way src/ply/StandardSplatDecoder.cpp does
python -m gstrain.cli verify C:\data\myscene\out\point_cloud\iteration_30000\point_cloud.ply

# time the rasterizer on this machine (no images needed)
python -m gstrain.cli bench -s C:\data\myscene -r 8
```

`-s` accepts either a COLMAP project root (containing `sparse/0/` and `images/`) or the
model folder itself. Binary `.bin` models are preferred over `.txt`, like COLMAP does.

Lens distortion is rectified on load for `SIMPLE_RADIAL`, `RADIAL` and `OPENCV` cameras,
so raw COLMAP output works without running `colmap image_undistorter` first. Fisheye and
`FULL_OPENCV` models are still rejected.

### Quick test

Run these from `training\`. Step 2 takes about 45 seconds on CPU.

```powershell
cd C:\Work\MayaGaussianSplatting\training

# 1. does the dataset parse? (cameras, images, points, scene extent, train/test split)
.\.venv\Scripts\python.exe -m gstrain.cli stats -s ..\gerrard-hall

# 2. deliberately tiny run: 1/24 resolution, 4 views, 5k seed points, 60 iterations
.\.venv\Scripts\python.exe -m gstrain.cli train -s ..\gerrard-hall -m ..\gerrard-hall\out `
    -r 24 --max-views 4 --max-points 5000 --iterations 60 `
    --log-every 10 --preview-every 30 --save-iterations 60 `
    --densify-from-iter 20 --densify-until-iter 50 --densification-interval 15

# 3. check the result decodes the way the Maya plugin will read it
.\.venv\Scripts\python.exe -m gstrain.cli verify ..\gerrard-hall\out\point_cloud.ply
```

It is working if all four of these hold:

| check | expected |
|---|---|
| progress bar `loss` | falls (roughly 0.46 -> 0.30) |
| progress bar `splats` | grows once densification starts (5000 -> ~14000) |
| `out\status.json` | `"state": "done"` |
| `out\point_cloud.ply` | exists and `verify` prints ~14000 splats, 62 properties |

Open `out\previews\latest.png` to see it: the render is on the left, the real photo on the
right. After 60 iterations it is dark and blurry, but the building must already line up
with the photo — that is what proves the camera poses, undistortion and rasterizer agree.

For an actual reconstruction drop the `--max-*` caps and raise the iteration count:

```powershell
.\.venv\Scripts\python.exe -m gstrain.cli train -s ..\gerrard-hall -m ..\gerrard-hall\out `
    -r 8 --iterations 7000 --device cuda
```

### Training notes

The rasterizer is pure PyTorch and runs on CPU or CUDA (`--device cuda`, the default `auto`
picks the GPU when present). Blending walks each tile in depth blocks, so peak memory is
bounded by `--element-budget` rather than by how crowded the busiest tile is.

Measured on gerrard-hall, RTX 2000 Ada vs a CPU thread pool, forward + backward:

| device | resolution | splats | per iteration |
|---|---|---|---|
| cpu | 351x234 (`-r 16`) | 20 000 | 852 ms |
| cuda | 351x234 (`-r 16`) | 20 000 | 159 ms |
| cuda | 1404x936 (`-r 4`) | 43 188 | 773 ms |

Use `gstrain.cli bench` to measure your own machine. `-r 2` and above is beyond what this
backend handles comfortably; `-r 4` is a sensible ceiling.

Useful knobs: `--eval-hold` (held-out views, default every 8th), `--preview-every`,
`--densify-until-iter`, `--densify-grad-threshold`, `--sh-degree-interval`, `--background`,
`--resume`, `--quiet`. `--up-axis maya-y` bakes a Y-up rotation into the exported `.ply`.

### What the console shows

Training reports each startup stage before it begins, so nothing looks hung — loading a
hundred full-size photos legitimately takes minutes:

```
gstrain: ..\gerrard-hall
     -> ..\gerrard-hall\out
[1/4] reading COLMAP model
      100 images, 43188 points, scene extent 5.057
[2/4] loading images (100 at 702x468)
      lens distortion rectified on load
      87 train / 13 held out, 142.6s
[3/4] initialising gaussians
      43188 splats, sh degree 3
[4/4] training 7000 iterations on cuda
      702x468, densify 500-15000 every 100, preview every 500
      iter 1000: sh degree -> 1
      iter 500: preview -> previews/iter_000500
 14%|▌   | 980/7000 [03:12<19:41, 5.09it/s, loss=0.0871, psnr=21.4, splats=98214]
```

`--quiet` silences all of it, for running as a subprocess. Progress is always written to
`progress.jsonl` regardless.

## Output layout

```
<model_dir>/
  point_cloud.ply                             # newest export, load this one in Maya
  point_cloud/iteration_<N>/point_cloud.ply   # per checkpoint
  progress.jsonl                              # one JSON object per logged iteration
  status.json                                 # running | done | error
  chkpnt<N>.pt                                # --checkpoint-every, resumable
  previews/iter_<N>/{view_XX,contact_sheet}.png + metrics.json
  previews/{latest,timeline}.png
```

## Tests

```powershell
python -m pytest -q
```

`tests/test_no_maya_imports.py` enforces the no-Maya rule (AST scan plus a clean
subprocess import). All tests build their own synthetic COLMAP model, so no dataset
is required.

| | |
|---|---|
| format | `binary_little_endian 1.0` (big endian is rejected by the loader) |
| element | `vertex N` |
| properties | `x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3`, all `float` |
| values | **raw / pre-activation** — the loader applies `exp` to scale, `sigmoid` to opacity, `0.5 + 0.28209479177387814 * f_dc` to colour |
| quaternion | `rot_0` is **W**, then X, Y, Z |
| `f_rest` | channel-major (15 R, 15 G, 15 B); ignored by this plugin, used by other viewers |

