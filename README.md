# AlignBeat

Code for **AlignBeat: A Latent Variable Model for Multi-Class Beat Tracking from
Partially Labeled Data** (Jaehoon Ahn, Tae Gum Hwang, Moon-Ryul Jung).
[arXiv:2510.14391](https://arxiv.org/abs/2510.14391)

Most recent beat trackers predict beats and downbeats with two independent
frame-wise binary heads, then reconcile them when they disagree. AlignBeat instead
predicts a sparse set of grid points. Each grid point has a continuous event time and
a single distribution over {downbeat, beat, no event}, so a downbeat is a beat by
construction. Which grid point accounts for which annotated event is a latent
alignment, fitted by hard expectation–maximization: the E-step finds the best
order-preserving alignment with an O(MN) dynamic program, and the M-step takes a
gradient step at that alignment. For datasets that annotate beats but not downbeats,
the event class is latent too and is marginalized out, so those datasets can be
kept in training instead of being dropped or masked.

Over 8-fold cross-validation on the 18 Beat This! datasets, with no post-processing
at any stage, downbeat CMLt rises 3.6 points over Beat This! and beat CMLt rises 3.1,
while both F-measures move by less than one point.

## Layout

AlignBeat builds on [Beat This!](https://github.com/CPJKU/beat_this) (Foscarin et
al., 2024), installed from PyPI as `beat-this==1.1.0`. Its frontend, transformer, data
pipeline and evaluation are reused unchanged. The `alignbeat/` package subclasses
them, replacing only the output head, the loss and the decoding.

| path | what |
| --- | --- |
| `alignbeat/model/head.py` | grid-point head: class head, time head, eq. (1) |
| `alignbeat/model/downsample.py` | strided-convolution reduction from frames to N grid points |
| `alignbeat/training/dp.py` | the E-step alignment, recursion (3) |
| `alignbeat/training/criterion.py` | the matching cost (2) and the M-step loss (4)–(5) |
| `alignbeat/training/priors.py` | per-fold statistics read off the training split |
| `alignbeat/inference/decode.py` | thresholding (6) and conversion to beat/downbeat times |
| `alignbeat/inference/stitching.py` | covering a full piece with overlapping fragments (§2.3) |
| `alignbeat/model/tracker.py` | Beat This!'s network with the grid-point head in place of its own |
| `alignbeat/integration/lightning.py` | Beat This!'s Lightning module, routed to the alignment head |
| `alignbeat/integration/data.py` | Beat This!'s data module, plus each piece's beat period |
| `launch_scripts/train.py` | Beat This!'s training script, plus `--head_type` and the flags below |
| `launch_scripts/compute_paper_metrics.py` | scoring over whole pieces |

Equation and section numbers refer to arXiv v3.

`--head_type dense` (the default) trains the original Beat This! model, and
`--head_type subset` trains AlignBeat. Everything else, including data, augmentation
and evaluation, is shared between the two.

## Setup

```sh
conda env create -f environment.yml
conda activate alignbeat
```

This installs the exact versions the paper's results were produced with (pinned in
`requirements.txt`), including the CUDA 11.8 build of PyTorch. Without conda,
`pip install -e ".[train]"` installs the same packages unpinned.

Data goes under `data/`, prepared as described in the
[Beat This! repository](https://github.com/CPJKU/beat_this). The folds are Beat This!'s
own `8-folds.split`, and `--fold N` names the fold held out.

## Training

As in the paper, the encoder starts from Beat This!'s released checkpoint for the
same fold, and the full model is trained for 20 more epochs:

```sh
curl -LO https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/fold0.ckpt

python launch_scripts/train.py --name alignbeat_f0 --head_type subset --fold 0 \
    --init_encoder_from fold0.ckpt --max-epochs 20
```

Run `python launch_scripts/train.py --help` for every flag. These are the
AlignBeat-specific ones:

| flag | effect |
| --- | --- |
| `--drop_beat_only` | leave out the beat-only datasets (Simac, SMC): the DROP arm of §4.1 |
| `--gamma` | weight γ of the background term in (5) (default 1) |
| `--detect_tau` | threshold τ in (6) (default 0.5) |
| `--attention_layers` | transformer layers at the start of the class head (default 2) |
| `--bpm_max`, `--train_length` | together these fix N, the number of grid points |
| `--downbeat_dropout` | treat this fraction of downbeat-annotated pieces as beat-only |

Some flags, such as `--latent_bar`, `--tempo_relative_b` and `--onset_adjustment`,
switch on extensions that go beyond the paper. They are kept for follow-up work and
do not affect the results reported there.

## Evaluation

```sh
python launch_scripts/compute_paper_metrics.py --models checkpoints/alignbeat_f0*.ckpt
```

## Tests

```sh
pytest tests
```

## History

This project began as BeatFCOS, a 1-D adaptation of FCOS with anchors, interval
regression and Soft-NMS (earlier versions of
[arXiv:2510.14391](https://arxiv.org/abs/2510.14391)). None of that anchor machinery
remains here.

## Citation

```bibtex
@misc{ahn2025alignbeat,
  title         = {{AlignBeat}: A Latent Variable Model for Multi-Class Beat Tracking
                   from Partially Labeled Data},
  author        = {Ahn, Jaehoon and Hwang, Tae Gum and Jung, Moon-Ryul},
  year          = {2025},
  eprint        = {2510.14391},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
}
```

Please also cite [Beat This!](https://github.com/CPJKU/beat_this), whose model,
data pipeline and evaluation this code builds on.

## License

MIT, the same as upstream Beat This!. See `LICENSE`.
