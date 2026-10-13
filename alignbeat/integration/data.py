"""Beat This!'s data pipeline, plus the one per-piece field the alignment head reads."""
import numpy as np
from beat_this.dataset import BeatDataModule
from beat_this.dataset.dataset import BeatTrackingDataset


class AlignBeatDataset(BeatTrackingDataset):
    """Adds "beat_period" to every item: the piece's median beat period in seconds.

    It describes the whole performance, not the sampled 30 s excerpt, and is measured
    after the tempo augmentation so it matches the stretched annotations. Median,
    because asap's rubato pulls a mean. 0.0 when a piece has under two beats, which
    the criterion reads as "fall back to the fixed scale" rather than dividing by zero.
    """

    def _get_spect(self, item):
        # Upstream's __getitem__ passes the augmented item here and nowhere else we
        # can reach, so the period is measured here and attached in __getitem__.
        beat_times = np.asarray(item["beat_time"], dtype=np.float64)
        period = float(np.median(np.diff(beat_times))) if beat_times.size >= 2 else 0.0
        self._beat_period = period if np.isfinite(period) and period > 0.0 else 0.0
        return super()._get_spect(item)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if isinstance(item, dict):
            item["beat_period"] = self._beat_period
        return item


class AlignBeatDataModule(BeatDataModule):
    """Upstream's datamodule, serving AlignBeatDataset for every split."""

    def setup(self, stage):
        super().setup(stage)
        for split in ("train", "val", "test", "predict"):
            dataset = getattr(self, f"{split}_dataset", None)
            if type(dataset) is BeatTrackingDataset:
                dataset.__class__ = AlignBeatDataset
