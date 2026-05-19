<h1 align="center">GeoRiceNet</h1>

<p align="center">
  <b>Reflection-aware geometry-gated density counting for UAV crop imagery</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-research%20code-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Data" src="https://img.shields.io/badge/Data-not%20included-6B7280?style=flat-square">
</p>

<p align="center">
  <img src="Network%20architecture%20of%20GeoRiceNet.png" alt="Network architecture of GeoRiceNet" width="960">
</p>

GeoRiceNet is a PyTorch implementation of a reflection-aware geometry-gated density counting model for UAV crop imagery. The release contains the model, prior construction utilities, losses, training entry points, and transfer-friendly reliability gate modules. Data, annotations, trained weights, generated experiment outputs, and manuscript files are intentionally excluded.

## Visual Overview

GeoRiceNet treats pseudo geometry and reflection priors as reliability cues rather than replacement modalities. The feature-level GeoGate calibrates RiceNet density features, and the transferable reliability gate can fuse RGB and geometry-guided density responses.

<p align="center">
  <img src="Comparison%20on%20public%20agricultural%20counting%20datasets.png" alt="Comparison on public agricultural counting datasets" width="820">
</p>

<p align="center">
  <sub><b>Qualitative comparison on public agricultural counting datasets.</b></sub>
</p>

## Demo Video

<p align="center">
  <video src="assets/demo.mp4" controls width="900"></video>
</p>

<p align="center">
  <a href="assets/demo.mp4">Open demo video</a>
</p>

Full-resolution figure sources:

- [Network architecture of GeoRiceNet](<Network architecture of GeoRiceNet.png>)
- [Comparison on public agricultural counting datasets](<Comparison on public agricultural counting datasets.png>)

## Repository Layout

```text
georicenet/
  baselines.py          Reference density-counting backbones
  datasets.py           Manifest-based image and density dataset
  gates.py              GeoGate and transferable reliability gate
  losses.py             Counting, density, structure, and gate losses
  model.py              GeoRiceNet wrapper around RiceNet
  priors.py             RGB, reflection, water, and geometry priors
  ricenet.py            RiceNet-style density baseline
scripts/
  train_georicenet.py   Training entry point for GeoRiceNet
  train_transfer_gate.py
requirements.txt
```

## What Is Not Included

The public repository intentionally excludes:

- raw UAV images and public benchmark copies
- point annotations, density maps, and HDF5 label files
- trained checkpoints and pretrained weights
- intermediate tables, logs, and temporary experiment outputs
- manuscript, revision, and similarity-check files

Use the CLI arguments in `scripts/` to point to your own data and checkpoints locally.

## Manifest Format

Training scripts expect CSV manifests with these columns:

```text
image_path,density_path,count
```

Paths may be absolute or relative to the manifest file. `density_path` should point to either:

- `.npy` density map, or
- `.h5` / `.hdf5` file containing a `density` dataset.

An optional `depth_path` column can be added when pseudo-depth maps are available:

```text
image_path,density_path,depth_path,count
```

If `depth_path` is omitted or empty, GeoRiceNet still runs and builds the reflection/texture priors from RGB alone.

The model predicts a density map, and the count is obtained by summing all density pixels.

## Quick Start

```bash
pip install -r requirements.txt

python scripts/train_georicenet.py \
  --train-manifest path/to/train.csv \
  --val-manifest path/to/val.csv \
  --output-dir outputs/georicenet \
  --epochs 20
```

To train only the transferable density-level reliability gate from cached branch outputs:

```bash
python scripts/train_transfer_gate.py \
  --train-cache path/to/train_cache \
  --val-cache path/to/val_cache \
  --output-dir outputs/transfer_gate
```

Each cache item should be an `.npz` file with:

```text
rgb_prior, geometry_prior, rgb_density, geo_density, target_density, count
```

## Notes

GeoRiceNet treats pseudo geometry as a reliability cue rather than a replacement for RGB. The feature-level GeoGate calibrates the density feature, while the transferable reliability gate can fuse any RGB backbone density response with a geometry-guided density response:

```text
D_out(x) = (1 - W(x)) * D_rgb(x) + W(x) * D_geo(x)
```

All file paths are supplied at runtime to keep the release independent from local experiment directories.
