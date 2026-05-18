# GeoRiceNet

GeoRiceNet is a PyTorch implementation of a reflection-aware geometry-gated density counting model for UAV crop imagery. The code release contains the model, prior construction utilities, losses, and transfer-friendly reliability gate modules. Data, annotations, trained weights, generated results, and manuscript files are not included.

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
- generated figures, tables, logs, and temporary experiment outputs
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
