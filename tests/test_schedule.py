import math

from supermimic.training.schedule import cosine_lr


def test_cosine_default_matches_paper_table():
    # Peak at 0, 3% from 250k/320k on, cosine in between.
    assert cosine_lr(0, 320_000, 1.0) == 1.0
    assert math.isclose(cosine_lr(125_000, 320_000, 1.0), 0.515, abs_tol=1e-3)
    assert cosine_lr(250_000, 320_000, 1.0) == 0.03
    assert cosine_lr(319_999, 320_000, 1.0) == 0.03


def test_cosine_anneal_from_resume_point():
    # A run resumed at 64k is annealed over 40k iterations: base lr before the start, 3% at the end.
    kw = dict(final_ratio=0.03, decay_fraction=1.0, start=64_000)
    assert cosine_lr(0, 104_000, 1.0, **kw) == 1.0
    assert cosine_lr(64_000, 104_000, 1.0, **kw) == 1.0
    assert math.isclose(cosine_lr(84_000, 104_000, 1.0, **kw), 0.515, abs_tol=1e-3)
    assert cosine_lr(104_000, 104_000, 1.0, **kw) == 0.03
    lrs = [cosine_lr(s, 104_000, 1.0, **kw) for s in range(64_000, 104_001, 1000)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
