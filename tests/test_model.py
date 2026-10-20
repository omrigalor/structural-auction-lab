import numpy as np
from model import recover_values, reserve_revenue, simulate_auctions


def test_inversion_moves_bids_toward_true_values():
    sample = simulate_auctions(2000, bidders=4, seed=4)
    recovered = recover_values(sample.bids, bidders=4)
    assert recovered.mean() > sample.bids.mean()
    assert abs(np.median(recovered) - 0.5) < 0.12


def test_revenue_is_nonnegative():
    sample = simulate_auctions(500, seed=2)
    recovered = recover_values(sample.bids, 4)
    assert reserve_revenue(recovered, 0.5) >= 0
