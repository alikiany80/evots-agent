import numpy as np
import pytest
from evots.metrics import score_boundaries, boundaries


def test_maximum_matching_not_nearest_greedy():
    r=score_boundaries([10,20],[1,11],100,10)
    assert r["tp"]==2 and r["f1"]==1


def test_inclusive_tolerance_and_no_double_counting():
    r=score_boundaries([50],[40,49,51],100,10)
    assert r["tp"]==1 and r["fp"]==2 and r["recall"]==1


@pytest.mark.parametrize("bad",[[0],[100],[1,1],[1.2],[float('nan')],[True],["12"]])
def test_bad_boundaries(bad):
    with pytest.raises(ValueError): boundaries(bad,100)


def test_empty_conventions_and_hausdorff():
    assert score_boundaries([],[],100)["f1"]==1
    assert score_boundaries([50],[],100)["hausdorff"] is None
    assert score_boundaries([], [50],100)["f1"]==0
    assert score_boundaries([10,80],[12,50],100)["hausdorff"]==30


def test_matching_against_bruteforce_small_sets():
    from itertools import combinations
    rng=np.random.default_rng(3)
    def brute(a,b,t):
        if not a or not b: return 0
        return max(brute(a[1:],b,t),max([1+brute(a[1:],b[:j]+b[j+1:],t) for j,p in enumerate(b) if abs(a[0]-p)<=t] or [0]))
    for _ in range(70):
        a=sorted(rng.choice(np.arange(1,30),4,replace=False).tolist())
        b=sorted(rng.choice(np.arange(1,30),4,replace=False).tolist())
        assert score_boundaries(a,b,40,4)["tp"]==brute(a,b,4)
