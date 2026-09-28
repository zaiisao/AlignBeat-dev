CLASS_UNKNOWN = -1
CLASS_DOWNBEAT = 0
CLASS_BEAT = 1
CLASS_BACKGROUND = 2

NUM_CLASSES = 3

# The standard beat-tracking F-measure tolerance, in SECONDS
# References:
#   [1] "Evaluation Methods for Musical Audio Beat Tracking Algorithms" by Davies et al., 2009
F_MEASURE_TOLERANCE = 0.07

# The continuity metrics' phase tolerance, as a FRACTION of the inter-annotation
# interval. Both constants are radii, not widths: [1] section 3.6 writes them in the
# same breath as "the +-70ms window of the F-measure, and the +-17.5% window for the
# continuity-based approaches", and [1]'s condition (i) reads
#     a_j - theta*Delta_j < gamma_b < a_j + theta*Delta_j,   theta = 0.175
# so the window spans theta*Delta either side. mir_eval agrees: beat.f_measure accepts
# |ref - est| <= 0.07, and beat.continuity tests |min_difference / reference_interval|
# < 0.175, both against an already-unsigned difference.
#
# The two are not interchangeable: F_MEASURE_TOLERANCE is absolute, this one scales with
# tempo, and they coincide only at 150 BPM, where 0.175 of a 0.4 s beat is 0.07 s.
CONTINUITY_TOLERANCE = 0.175

# Fastest tempo the corpus contains, in BPM. Only an upper bound is needed:
# overshooting costs a few extra candidates that get classified as background, which
# the formulation expects anyway, while undershooting is unrecoverable -- an
# order-preserving injection needs N >= M, and SubsetCriterion skips any fragment where
# it does not hold. 15 of 5555 tracks exceed this, all of them asap, whose MIDI-derived
# annotations are note-level rather than beat-level (the densest has a 0.005 s gap,
# i.e. 11521 BPM); those are bad annotations, not fast music.
BPM_MAX = 340.0
