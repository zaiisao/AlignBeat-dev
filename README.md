# AlignBeat

Code for **AlignBeat: A Latent Variable Model for Multi-Class Beat Tracking from
Partially Labeled Data**, by Jaehoon Ahn, Tae Gum Hwang, and Moon-Ryul Jung.
[Preprint: arXiv:2510.14391v3](https://arxiv.org/abs/2510.14391v3).

AlignBeat adds a candidate event head to Beat This! for beat and downbeat tracking
as **latent order-preserving alignment**. The head regresses event times and
classifies candidates as downbeat, beat, or background. An O(NM) dynamic program
selects an ordered subset of candidates to match the annotated events. Beat-only
annotations can contribute to training while the downbeat labels are unobserved.

Some equation and algorithm references in source comments use the numbering of
earlier working drafts and may differ from the linked preprint.

## Setup and data

Create and activate the tested Linux CUDA environment from the repository root:

```sh
conda env create -f environment.yml
conda activate alignbeat
```

The environment includes the training, evaluation, preprocessing, and test
dependencies. It uses the PyTorch 2.7.1 CUDA 11.8 wheels; for CPU or a different
CUDA build, edit the `torch` and `torchaudio` entries in `environment.yml` to use
the matching wheels for your platform.

The distribution is named `alignbeat` and includes both local Python packages,
`alignbeat` and `beat_this`. The inherited `beat_this` command runs the upstream
Beat This! inference workflow; the commands below use the AlignBeat training and
evaluation paths.

`requirements.txt` retains upstream's inference dependency pins. It is not a
complete pinned environment for AlignBeat training; the `train` installation
extra declares the additional training and evaluation dependencies.

Prepare data under `data/` using the instructions in the
[Beat This! repository](https://github.com/CPJKU/beat_this#data). Cross-validation
uses its `8-folds.split` files; `--fold N` selects the held-out fold.

## Training

Run commands from the repository root. The paper initializes each fold's encoder
from the matching released Beat This! checkpoint, then trains AlignBeat for 20 epochs.
This example reproduces fold 0, using the default 1500-frame window at 50 frames per
second (30 seconds):

```sh
mkdir -p checkpoints
curl -L https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/fold0.ckpt \
    -o checkpoints/beat-this-fold0.ckpt
python launch_scripts/train.py --name alignbeat_f0 --fold 0 \
    --init_encoder_from checkpoints/beat-this-fold0.ckpt
```

For another cross-validation fold, download its matching `foldN.ckpt` and pass that
checkpoint with `--fold N`. The encoder is fine-tuned together with the AlignBeat
heads. To train the dense Beat This! baseline instead, use `--head_type dense`.

The current training CLI exposes the following settings. These are its defaults,
not a claim that every historical checkpoint used the same configuration.

| flag | default | effect |
| --- | --- | --- |
| `--head_type` | `subset` | selects the dense Beat This! head or AlignBeat's `subset` head |
| `--max-epochs` | `20` | maximum training epochs |
| `--attention_layers` | `2` | candidate attention layers in the classification branch |
| `--detect_tau` | `0.5` | threshold on candidate event probability |
| `--match_event_cost` | enabled | includes beat-only event probability in matching; `--no-match_event_cost` disables it |
| `--latent_bar` | disabled | enables an additional shared meter/phase marginalization for beat-only data |
| `--gamma` | `1.0` | background loss coefficient |
| `--tempo_relative_b` | disabled | uses a piece's beat period to scale the timing loss |
| `--onset_adjustment` | disabled | adds onset-strength calibration to the event probabilities |
| `--drop_beat_only` | disabled | excludes pieces without downbeat annotations |
| `--downbeat_dropout` | `0.0` | hides downbeat annotations on a fraction of training pieces |

The subset head's default loss settings follow the paper: `gamma=1.0`, active-event
mass in beat-only matching, and no latent meter/phase marginalization. The default
learning rate is `3e-4` for the subset head and `8e-4` for the dense head unless
`--lr` is supplied.

Run `python launch_scripts/train.py --help` for the complete option list.

## Evaluation

```sh
# Score one checkpoint on its validation split
python launch_scripts/compute_paper_metrics.py --models checkpoints/alignbeat_f0*.ckpt

# Aggregate checkpoints for all held-out folds
python launch_scripts/compute_paper_metrics.py --models checkpoints/alignbeat_f*.ckpt \
    --aggregation-type k-fold
```

Let the shell expand the checkpoint patterns. Use `--datasplit test` for the held-out
test set. `--read-out` can select `per_event`, `map`, or `duration` decoding;
`--stitch-border` overrides the fragment border. The script also accepts
`--dump-piece-metrics` and `--dump-predictions` for saving its outputs.

`tests/` contains mathematical checks and regression tests. Install test, training,
and preprocessing dependencies with
`pip install -e ".[test,train,preprocess]"`; run the quick suite with
`python -m pytest -m 'not integration and not slow'`. The pretrained inference
checks are marked `integration`, and the `torch.compile` check is marked `slow`.

## History

This began as BeatFCOS, a 1-D adaptation of FCOS with anchors, interval
regression and Soft-NMS ([arXiv:2510.14391](https://arxiv.org/abs/2510.14391)),
then moved from a WaveBeat backbone to Beat Transformer to Beat This!, and
finally dropped anchors entirely for the subset-selection formulation above.
None of the anchor machinery survives here; the original is preserved in its
own repository.

## License

MIT, matching upstream Beat This!. The upstream and AlignBeat copyright notices
are preserved in [LICENSE](LICENSE).
