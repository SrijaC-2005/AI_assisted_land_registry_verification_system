"""Consensus policy for the land registry: one place for every rule that is a
decision rather than a measurement.

Used by:
  app.py                      - per-transaction fault-tolerance requirement and
                                the safety guard applied to every ML prediction
  benchmark_consensus.py      - labelling benchmark scenarios
  train_consensus_selector.py - (indirectly) the labels it learns

All thresholds can be overridden with environment variables so an
administrator can change policy without editing code. Every change of policy
requires re-running the benchmark and retraining, because labels depend on it.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

PROTOCOLS = ("PoW", "PoS", "Raft", "PBFT", "HotStuff")

# ------------------------------------------------------------------ policy inputs
HIGH_VALUE_THRESHOLD_INR = float(os.environ.get("LAND_REGISTRY_HIGH_VALUE_INR", "5000000"))   # ₹50 lakh
FT_STANDARD = float(os.environ.get("LAND_REGISTRY_FT_STANDARD", "0.50"))   # crash-fault tolerance acceptable
FT_STRICT = float(os.environ.get("LAND_REGISTRY_FT_STRICT", "0.75"))       # Byzantine fault tolerance required
BFT_FT_THRESHOLD = 2.0 / 3.0            # requirement at/above this means BFT is mandatory
BFT_RISK_THRESHOLD = float(os.environ.get("LAND_REGISTRY_BFT_RISK", "0.30"))  # GNN risk (0-1) needing BFT
BFT_MIN_NODES = 4                       # n >= 3f + 1 with f >= 1

# Ties: measured latencies within this margin are treated as equal, and the
# protocol with fewer messages per round wins.
TIE_RELATIVE = 0.10
TIE_ABSOLUTE_MS = 0.5

EXCLUDED = {
    "PoW": "Excluded by policy: computational mining is unnecessary and energy-intensive in a permissioned government registry.",
    "PoS": "Excluded by policy: requires staking economics and slashing, which a government registry does not have; no Byzantine guarantee in this implementation.",
}


def fault_tolerance_requirement(land_price: Optional[float], risk_level: Optional[str]) -> Tuple[float, str]:
    """Configured policy value for one transaction, with the rule that produced it."""
    try:
        price = float(land_price) if land_price is not None else 0.0
    except (TypeError, ValueError):
        price = 0.0
    level = str(risk_level or "").upper()
    if price >= HIGH_VALUE_THRESHOLD_INR:
        return FT_STRICT, f"Policy: high-value transfer (≥ ₹{HIGH_VALUE_THRESHOLD_INR:,.0f}) requires Byzantine fault tolerance"
    if level in ("MEDIUM", "HIGH"):
        return FT_STRICT, f"Policy: {level} fraud-risk transfer requires Byzantine fault tolerance"
    return FT_STANDARD, "Policy: standard-value, low-risk transfer; crash-fault tolerance acceptable"


def bft_required(security_risk: float, ft_requirement: float) -> bool:
    return ft_requirement >= BFT_FT_THRESHOLD or security_risk >= BFT_RISK_THRESHOLD


def messages_per_round(protocol: str, n: int) -> int:
    """Approximate messages for one committed block (standard textbook counts)."""
    n = max(int(n), 1)
    return {
        "Raft": 2 * (n - 1),              # append entries + acknowledgements
        "HotStuff": 8 * (n - 1),          # 4 phases x (proposal + vote), linear
        "PBFT": (n - 1) + 2 * n * (n - 1),  # pre-prepare + all-to-all prepare and commit
        "PoS": 2 * (n - 1),
        "PoW": n - 1,
    }.get(protocol, 10**6)


def eligible_protocols(n: int, security_risk: float, ft_requirement: float) -> Dict[str, str]:
    """Return {protocol: reason} for protocols allowed by policy; excluded ones are omitted."""
    allowed: Dict[str, str] = {}
    need_bft = bft_required(security_risk, ft_requirement)
    if not need_bft:
        allowed["Raft"] = "Crash-fault tolerance is sufficient for this transaction."
    if n >= BFT_MIN_NODES:
        allowed["PBFT"] = f"Byzantine fault tolerant; {n} nodes tolerate f = {(n - 1) // 3}."
        allowed["HotStuff"] = f"Byzantine fault tolerant; {n} nodes tolerate f = {(n - 1) // 3}."
    return allowed


def ineligibility_reason(protocol: str, n: int, security_risk: float, ft_requirement: float) -> Optional[str]:
    """None if the protocol is allowed, otherwise the reason it is not."""
    if protocol in EXCLUDED:
        return EXCLUDED[protocol]
    if protocol in eligible_protocols(n, security_risk, ft_requirement):
        return None
    if protocol == "Raft":
        return ("Raft tolerates crashes only, but this transaction requires Byzantine fault tolerance "
                f"(fault-tolerance requirement {ft_requirement:.2f}, risk {security_risk:.2f}).")
    return f"{protocol} needs at least {BFT_MIN_NODES} nodes (n = 3f + 1); only {n} registered."


def choose_label(n: int, security_risk: float, ft_requirement: float,
                 latency_by_protocol: Dict[str, float]) -> Tuple[Optional[str], str]:
    """Benchmark label: the eligible protocol with the lowest measured latency.

    Latencies within TIE_RELATIVE / TIE_ABSOLUTE_MS are treated as equal and the
    tie is broken by fewer messages per round.
    """
    allowed = [p for p in eligible_protocols(n, security_risk, ft_requirement) if p in latency_by_protocol]
    if not allowed:
        return None, "No protocol satisfies the policy for this scenario."
    best_latency = min(latency_by_protocol[p] for p in allowed)
    margin = max(TIE_ABSOLUTE_MS, TIE_RELATIVE * best_latency)
    near_best = [p for p in allowed if latency_by_protocol[p] - best_latency <= margin]
    winner = min(near_best, key=lambda p: (messages_per_round(p, n), latency_by_protocol[p]))
    reason = (
        f"Eligible: {', '.join(allowed)}. "
        f"Within {margin:.2f} ms of best latency: {', '.join(near_best)}. "
        f"Chosen {winner}: fewest messages per round ({messages_per_round(winner, n)}) among those."
    )
    return winner, reason