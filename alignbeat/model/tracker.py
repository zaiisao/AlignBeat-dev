"""Beat This!'s network with its two frame-wise heads swapped for the grid-point head."""
from beat_this.model.beat_tracker import BeatThis

from alignbeat.model.head import SubsetHead, librosa_logmel_onset_strength


class AlignBeatTracker(BeatThis):
    """Upstream frontend and transformer, unchanged, followed by a SubsetHead.

    Takes upstream's constructor arguments plus the head's own. The parameter names
    match upstream's, so a state_dict saved by either loads into the other's encoder.
    """

    def __init__(self, subset_head_kwargs: dict, transformer_dim: int = 512, **kwargs):
        super().__init__(transformer_dim=transformer_dim, **kwargs)
        # Replacing the attribute keeps the "task_heads" key, so checkpoint keys and
        # parameter order are the same as when the head was built in place.
        self.task_heads = SubsetHead(transformer_dim, **subset_head_kwargs)
        self.task_heads.apply(self._init_weights)
        # ...then restore the head's own initialisation, which the generic pass above
        # overwrites: the zeroed regression and class weights.
        self.task_heads.head._initialize_weights()
        self.task_heads.initialize_event_calibrator()

    def forward(self, x):
        # torch.compile wraps a module in OptimizedModule, so read the head's own
        # flag through _orig_mod to keep the onset path in compiled runs too.
        task_heads = getattr(self.task_heads, "_orig_mod", self.task_heads)
        onset_strength = (librosa_logmel_onset_strength(x)
                          if task_heads.onset_adjustment else None)
        x = self.frontend(x)
        x = self.transformer_blocks(x)
        return self.task_heads(x, onset_strength=onset_strength)
