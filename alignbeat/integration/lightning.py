"""Beat This!'s Lightning module, routed to the alignment head when subset_head=True.

With subset_head=False every method falls through to upstream, so the same module
trains and scores the dense Beat This! baseline.
"""
import torch
from beat_this.model.pl_module import PLBeatThis

from alignbeat.integration.config import split_subset_kwargs
from alignbeat.integration.subset import subset_decode, subset_loss, subset_predict_piece
from alignbeat.model.tracker import AlignBeatTracker
from alignbeat.training.criterion import SubsetCriterion


class PLAlignBeat(PLBeatThis):
    # The signature repeats upstream's in full rather than taking **kwargs: scripts
    # filter checkpoint hyper_parameters against it, and save_hyperparameters reads
    # the arguments by name.
    def __init__(
        self,
        spect_dim=128,
        fps=50,
        transformer_dim=512,
        ff_mult=4,
        n_layers=6,
        stem_dim=32,
        dropout={"frontend": 0.1, "transformer": 0.2},
        lr=0.0008,
        weight_decay=0.01,
        pos_weights={"beat": 1, "downbeat": 1},
        head_dim=32,
        loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000,
        max_epochs=100,
        use_dbn=False,
        eval_trim_beats=5,
        sum_head=True,
        partial_transformers=True,
        subset_head=False,
        subset_kwargs=None,
    ):
        super().__init__(
            spect_dim=spect_dim, fps=fps, transformer_dim=transformer_dim,
            ff_mult=ff_mult, n_layers=n_layers, stem_dim=stem_dim, dropout=dropout,
            lr=lr, weight_decay=weight_decay, pos_weights=pos_weights,
            head_dim=head_dim, loss_type=loss_type, warmup_steps=warmup_steps,
            max_epochs=max_epochs, use_dbn=use_dbn, eval_trim_beats=eval_trim_beats,
            sum_head=sum_head, partial_transformers=partial_transformers,
        )
        self.subset_criterion = None
        self.decode_kwargs = None
        if not subset_head:
            return
        if subset_kwargs is None:
            raise ValueError("subset_head=True needs subset_kwargs")

        head_kwargs, self.decode_kwargs, criterion_kwargs = split_subset_kwargs(
            subset_kwargs)
        # Upstream has just built a dense BeatThis; replace it with the same encoder
        # under the alignment head.
        self.model = AlignBeatTracker(
            subset_head_kwargs=head_kwargs,
            spect_dim=spect_dim,
            transformer_dim=transformer_dim,
            ff_mult=ff_mult,
            stem_dim=stem_dim,
            n_layers=n_layers,
            head_dim=head_dim,
            dropout=dropout,
            sum_head=sum_head,
            partial_transformers=partial_transformers,
        )
        self.subset_criterion = SubsetCriterion(
            **criterion_kwargs, window_seconds=head_kwargs["train_length"] / fps)

    def _compute_loss(self, batch, model_prediction):
        if self.subset_criterion is None:
            return super()._compute_loss(batch, model_prediction)
        return subset_loss(self.subset_criterion, batch, model_prediction, self.fps)

    def log_losses(self, losses, batch_size, step="train"):
        # Upstream logs "beat" and "downbeat" by name; the alignment head's terms are
        # "class" and "time", so log whatever terms the loss returned.
        for target in (k for k in losses if k != "total"):
            self.log(
                f"{step}_loss_{target}",
                losses[target].item(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        self.log(
            f"{step}_loss",
            losses["total"].item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        if self.subset_criterion is None:
            return super().validation_step(batch, batch_idx)
        model_prediction = self.model(batch["spect"])
        losses = self._compute_loss(batch, model_prediction)
        postp_beat, postp_downbeat = subset_decode(
            self.subset_criterion, batch, model_prediction, self.fps,
            **self.decode_kwargs)
        metrics = self._compute_metrics(batch, postp_beat, postp_downbeat, step="val")
        self.log_losses(losses, len(batch["spect"]), "val")
        self.log_metrics(metrics, batch["spect"].shape[0], "val")

    def test_step(self, batch, batch_idx):
        if self.subset_criterion is None:
            return super().test_step(batch, batch_idx)
        # The alignment head returns no frame-wise prediction for a whole piece, so
        # there is no test loss to log, only metrics.
        metrics, _, _, _ = self.predict_step(batch, batch_idx)
        self.log_metrics(metrics, batch["spect"].shape[0], "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0, chunk_size=1500,
                     overlap_mode="keep_first"):
        if self.subset_criterion is None:
            return super().predict_step(batch, batch_idx, dataloader_idx, chunk_size,
                                        overlap_mode)
        if batch["spect"].shape[0] != 1:
            raise ValueError(
                "When predicting full pieces, only `batch_size=1` is supported")
        if torch.any(~batch["padding_mask"]):
            raise ValueError(
                "When predicting full pieces, the Dataset must not pad inputs")
        beats, downbeats = subset_predict_piece(
            self.model, self.subset_criterion, batch, chunk_size, self.fps,
            **self.decode_kwargs)
        metrics = self._compute_metrics(batch, beats, downbeats, step="test")
        # The decoded times themselves, not dense activations: this head has no
        # postprocessor to run, so --dump-predictions consumes these as they are.
        # batch_size is 1 here, so each is a 1-tuple holding one piece's array.
        return (metrics, (beats[0], downbeats[0]),
                batch["dataset"], batch["spect_path"])
