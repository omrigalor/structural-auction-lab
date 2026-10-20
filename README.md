# Structural Auction Lab

An econometrics sandbox for first-price sealed-bid auctions: equilibrium simulation, GPV-style nonparametric inversion from bids to latent private values, and counterfactual reserve-price revenue.

```python
from model import simulate_auctions, recover_values, reserve_revenue
sample = simulate_auctions(n_auctions=2_000, bidders=4)
values = recover_values(sample.bids, bidders=4)
print(reserve_revenue(values, reserve=0.5))
```

The inversion implements the first-order-condition idea `v = b + G(b)/((n−1)g(b))`, estimating the bid CDF empirically and the density with a Gaussian kernel.

## Boundaries

Identification depends on independent private values, symmetry, risk neutrality, monotone bidding, and correct bidder counts. Boundary bias and bandwidth choice materially affect pseudo-values. Counterfactuals are illustrative and should not be used for real market design without a fuller specification and robustness analysis.

Requires NumPy. Tests use `pytest -q`.
