"""Flat-forward vol term structure signal."""

from __future__ import annotations

from data.chain import OptionChain


def flat_forward_signal(near_chain: OptionChain, far_chain: OptionChain) -> float:
    """Near/far ATM IV ratio. > 1 suggests near-term vol premium."""
    far_iv = far_chain.atm_iv()
    if far_iv == 0:
        return 0.0
    return near_chain.atm_iv() / far_iv
