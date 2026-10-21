"""Owner title verification and rule-based owner trust score.

Two separate things, both deterministic and auditable:

1. verify_title(land_id, seller, history)
   A registry must not let someone transfer land they do not own. If the land
   already has a recorded transfer (ledger or pending queue), the seller must
   be the most recent recorded owner. Land with no prior record is treated as
   a first registration and flagged as "prior title not on this ledger".

2. compute_owner_trust(seller, land_id, history, blocked)
   A 0-100 score built from documented rules, with a line-by-line breakdown.
   It is NOT a machine-learning output and is NOT an input to the GNN or the
   consensus selector. It exists so an officer can see, at a glance, the
   seller's record on this registry.

Rules (each line appears in the breakdown):
   Base                                                     50
   Title verified (seller is current recorded owner)       +20
   First registration (no prior record for this land)       +0  (flagged)
   Completed transfers as seller on the ledger          +4 each, max +20
   Transfers blocked by fraud screening                -15 each, max -45
   Average GNN risk of seller's past transactions     -0.3 x avg%, max -30
   Clamped to 0..100.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_SCORE = 50.0
TITLE_VERIFIED_BONUS = 20.0
COMPLETED_PER_TX = 4.0
COMPLETED_MAX = 20.0
BLOCKED_PER_TX = -15.0
BLOCKED_MAX = -45.0
RISK_FACTOR = -0.3
RISK_MAX = -30.0


def _norm(name: Any) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _sort_key(tx: Dict[str, Any]) -> Tuple[int, str]:
    # Ledger order first (block index), then pending order by ISO timestamp.
    idx = tx.get("block_index")
    return (idx if isinstance(idx, int) else 10**9, str(tx.get("timestamp", "")))


def land_ownership_chain(land_id: str, history: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All recorded transfers of one land parcel, oldest first."""
    target = _norm(land_id)
    chain = [tx for tx in history if isinstance(tx, dict) and _norm(tx.get("land_id")) == target]
    return sorted(chain, key=_sort_key)


def verify_title(land_id: str, seller: str, history: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    chain = land_ownership_chain(land_id, history)
    if not chain:
        return {
            "status": "first_registration",
            "verified": False,
            "allowed": True,
            "current_owner": None,
            "message": "No prior record of this land on the registry. Treated as first registration; "
                       "prior title must be verified from offline records.",
        }
    current_owner = chain[-1].get("new_owner")
    if _norm(current_owner) == _norm(seller):
        return {
            "status": "verified",
            "verified": True,
            "allowed": True,
            "current_owner": current_owner,
            "message": f"Seller is the current recorded owner (from {len(chain)} prior record(s)).",
        }
    return {
        "status": "owner_mismatch",
        "verified": False,
        "allowed": False,
        "current_owner": current_owner,
        "message": f"Transfer refused: the registry records '{current_owner}' as the current owner "
                   f"of land {land_id}, not '{seller}'.",
    }


def compute_owner_trust(
    seller: str,
    land_id: str,
    history: Iterable[Dict[str, Any]],
    blocked: Iterable[Dict[str, Any]],
    title: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    history = list(history)
    seller_n = _norm(seller)
    title = title or verify_title(land_id, seller, history)

    breakdown: List[Dict[str, Any]] = [{"rule": "Base score", "points": BASE_SCORE}]

    if title["status"] == "verified":
        breakdown.append({"rule": "Title verified: seller is the current recorded owner",
                          "points": TITLE_VERIFIED_BONUS})
    elif title["status"] == "first_registration":
        breakdown.append({"rule": "First registration: prior title not on this ledger (needs offline check)",
                          "points": 0.0})

    completed = [tx for tx in history
                 if tx.get("source") == "ledger" and _norm(tx.get("old_owner")) == seller_n]
    pts = min(COMPLETED_MAX, COMPLETED_PER_TX * len(completed))
    breakdown.append({"rule": f"{len(completed)} completed transfer(s) as seller on the ledger", "points": pts})

    blocked_list = []
    for item in blocked:
        tx = item.get("transaction") if isinstance(item, dict) else None
        if isinstance(tx, dict) and _norm(tx.get("old_owner")) == seller_n:
            blocked_list.append(tx)
    pts = max(BLOCKED_MAX, BLOCKED_PER_TX * len(blocked_list))
    breakdown.append({"rule": f"{len(blocked_list)} transfer(s) blocked by fraud screening", "points": pts})

    risks = []
    for tx in history:
        if _norm(tx.get("old_owner")) != seller_n:
            continue
        score = (tx.get("risk_assessment") or {}).get("risk_score")
        try:
            risks.append(float(score))
        except (TypeError, ValueError):
            pass
    if risks:
        avg = sum(risks) / len(risks)
        pts = max(RISK_MAX, RISK_FACTOR * avg)
        breakdown.append({"rule": f"Average GNN risk of {len(risks)} past transaction(s): {avg:.1f}%",
                          "points": round(pts, 2)})
    else:
        breakdown.append({"rule": "No past transactions with a GNN risk score", "points": 0.0})

    total = max(0.0, min(100.0, sum(b["points"] for b in breakdown)))
    level = "HIGH" if total >= 70 else ("MEDIUM" if total >= 40 else "LOW")
    return {
        "score": round(total, 1),
        "level": level,
        "method": "Rule-based (documented in owner_trust.py); not a machine-learning output",
        "title_status": title["status"],
        "breakdown": breakdown,
    }