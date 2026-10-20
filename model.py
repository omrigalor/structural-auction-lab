"""Simulation and GPV-style inversion for first-price sealed-bid auctions."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AuctionSample:
    values: np.ndarray
    bids: np.ndarray


def simulate_auctions(n_auctions=1000, bidders=4, seed=0):
    """Symmetric risk-neutral equilibrium for Uniform(0,1) private values."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(size=(n_auctions, bidders))
    bids = (bidders - 1) / bidders * values
    return AuctionSample(values, bids)


def recover_values(bids, bidders, bandwidth=None):
    """Recover pseudo-values v=b+G(b)/((n-1)g(b)) via empirical CDF and Gaussian KDE."""
    b = np.asarray(bids, float).ravel()
    if bidders < 2 or b.size < 10:
        raise ValueError("need at least two bidders and ten bids")
    h = bandwidth or 1.06 * np.std(b, ddof=1) * b.size ** (-1 / 5)
    h = max(float(h), 1e-4)
    z = (b[:, None] - b[None, :]) / h
    density = np.exp(-0.5 * z**2).mean(axis=1) / (h * np.sqrt(2 * np.pi))
    ranks = np.argsort(np.argsort(b))
    cdf = (ranks + 0.5) / b.size
    return b + cdf / ((bidders - 1) * np.maximum(density, 1e-8))


def reserve_revenue(pseudo_values, reserve, bidders=4, draws=20_000, seed=0):
    """Bootstrap estimated private values and evaluate second-price reserve revenue."""
    rng = np.random.default_rng(seed)
    values = rng.choice(np.asarray(pseudo_values), size=(draws, bidders), replace=True)
    values.sort(axis=1)
    highest, second = values[:, -1], values[:, -2]
    return float(np.where(highest >= reserve, np.maximum(reserve, second), 0).mean())
