"""Black-Scholes pricing and implied volatility solver."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        sigma: Volatility.
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put via put-call parity."""
    if T <= 0:
        return max(K - S, 0.0)
    return bs_call_price(S, K, T, r, sigma) - S + K * math.exp(-r * T)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega: sensitivity of option price to volatility."""
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * math.sqrt(T) * _norm_pdf(d1)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CE",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float | None:
    """Newton-Raphson implied volatility solver.

    Args:
        market_price: Observed option price.
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        option_type: "CE" or "PE".
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Implied volatility or None if solver fails to converge.
    """
    if T <= 0 or market_price <= 0:
        return None

    price_fn = bs_call_price if option_type == "CE" else bs_put_price

    # Initial guess via Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / T) * market_price / S

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            return None
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.001

    return None


def realized_vol(prices: pd.Series, window: int = 30) -> pd.Series:
    """Annualized realized volatility from close prices.

    Args:
        prices: Close price series.
        window: Rolling window in trading days.

    Returns:
        Series of annualized realized vol.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252)


def compute_historical_iv_series(
    option_prices: pd.Series,
    underlying_prices: pd.Series,
    strike: float,
    expiry_date: pd.Timestamp,
    r: float,
    option_type: str = "CE",
) -> pd.Series:
    """Compute IV for each historical date from option close prices.

    Args:
        option_prices: Daily option close prices.
        underlying_prices: Daily underlying close prices (aligned index).
        strike: Option strike price.
        expiry_date: Expiry date for T calculation.
        r: Risk-free rate.
        option_type: "CE" or "PE".

    Returns:
        Series of implied volatilities indexed by date.
    """
    ivs = {}
    for date in option_prices.index:
        T = (expiry_date - date).days / 365.0
        if T <= 0:
            continue
        S = underlying_prices.get(date)
        price = option_prices.get(date)
        if S is None or price is None or np.isnan(S) or np.isnan(price):
            continue
        iv = implied_vol(float(price), float(S), strike, T, r, option_type)
        if iv is not None:
            ivs[date] = iv
    return pd.Series(ivs, name="iv")
