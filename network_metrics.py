"""Runtime measurements for the local Digital Land Registry prototype.

What is measured, and what it is NOT:

- node_count
    Number of nodes registered in the in-process Network object.

- network_latency_ms  (feature name kept for model compatibility)
    Median duration of recent consensus rounds (block forging + commit),
    measured inside Network.run_consensus(). This is an *in-process consensus
    round latency*. It is NOT peer-to-peer RTT, because the prototype's nodes
    are simulated inside one Python process.

    Earlier versions used the full Flask request duration here. That included
    PDF hashing, PQC signing, GNN fraud scoring and SHAP, which is why values
    such as 12,437 ms appeared. Request duration is still recorded, but only as
    a diagnostic measurement; it is no longer fed to the consensus selector.

- tx_throughput_tps
    Transactions committed to the ledger in the rolling window, divided by the
    window length. This is application commit throughput, not distributed
    network throughput.

- security_risk_score
    Current transaction's GNN fraud risk score (percent) converted to 0-1.

- fault_tolerance_requirement
    Configured policy value. Not measured.

Energy is not reported because no power telemetry is connected.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from statistics import median
from threading import Lock
from time import time
from typing import Any, Dict, List, Optional
import csv
import json
import logging

logger = logging.getLogger(__name__)

FEATURE_SOURCE_TYPES = {
    "node_count": "measured",
    "network_latency_ms": "measured (in-process consensus round; proxy, not P2P RTT)",
    "tx_throughput_tps": "measured (application commit rate)",
    "security_risk_score": "calculated (GNN fraud risk, % -> 0-1)",
    "fault_tolerance_requirement": "configured policy",
}

OBSERVATION_COLUMNS = [
    "timestamp_epoch",
    "node_count",
    "network_latency_ms",
    "tx_throughput_tps",
    "security_risk_score",
    "fault_tolerance_requirement",
    "consensus_rounds_in_window",
    "committed_transactions_in_window",
    "window_seconds",
    "request_processing_latency_ms",
    "active_consensus_at_measurement",
    "out_of_training_range_features",
    "input_sources_json",
    "consensus",  # benchmark label; left blank on purpose
]


class RuntimeMetrics:
    def __init__(
        self,
        window_seconds: float = 300.0,
        observations_path: Optional[str] = None,
    ) -> None:
        self.window_seconds = float(window_seconds)
        default_path = Path(__file__).resolve().parent / "Dataset" / "Runtime_Measured_Metrics.csv"
        self.observations_path = Path(observations_path) if observations_path else default_path
        self._lock = Lock()
        self._observation_lock = Lock()
        # (timestamp, duration_ms, committed_tx_count, protocol, kind)
        self._consensus_rounds: deque = deque(maxlen=500)
        # (timestamp, duration_ms) -- diagnostic only
        self._request_latencies: deque = deque(maxlen=500)

    # ------------------------------------------------------------------ record
    def record_consensus_round(
        self,
        duration_ms: float,
        committed_transactions: int,
        protocol: str,
        timestamp: Optional[float] = None,
        kind: str = "block",
    ) -> None:
        """Record one timed consensus round.

        kind='block'  real block forged and committed (counts toward throughput)
        kind='probe'  dry-run of the active protocol, nothing committed
        """
        if duration_ms is None or float(duration_ms) < 0:
            return
        stamp = time() if timestamp is None else float(timestamp)
        with self._lock:
            self._consensus_rounds.append(
                (stamp, float(duration_ms), int(committed_transactions), str(protocol), str(kind))
            )

    def record_request_latency(self, latency_ms: float, timestamp: Optional[float] = None) -> None:
        """Full HTTP request duration. Diagnostic only; never a model input."""
        if latency_ms is None or float(latency_ms) < 0:
            return
        stamp = time() if timestamp is None else float(timestamp)
        with self._lock:
            self._request_latencies.append((stamp, float(latency_ms)))

    # Backwards-compatible name used by older app.py versions.
    def record_processing_latency(self, latency_ms: float, timestamp: Optional[float] = None) -> None:
        self.record_request_latency(latency_ms, timestamp)

    def record_successful_transaction(self, timestamp: Optional[float] = None) -> None:
        """Kept for compatibility. Throughput now counts committed transactions."""
        return None

    # ---------------------------------------------------------------- snapshot
    def _window(self, now: float):
        cutoff = now - self.window_seconds
        rounds = [r for r in self._consensus_rounds if r[0] >= cutoff]
        requests = [r for r in self._request_latencies if r[0] >= cutoff]
        return rounds, requests

    def has_recent_round(self, now: Optional[float] = None) -> bool:
        stamp = time() if now is None else float(now)
        with self._lock:
            rounds, _ = self._window(stamp)
        return bool(rounds)

    def snapshot(
        self,
        active_node_count: int,
        fraud_risk_score_pct: Optional[float],
        fault_tolerance_requirement: float = 0.75,
        active_consensus: Optional[str] = None,
        fault_tolerance_source: Optional[str] = None,
        now: Optional[float] = None,
        append_observation: bool = True,
    ) -> Dict[str, Any]:
        stamp = time() if now is None else float(now)
        with self._lock:
            rounds, requests = self._window(stamp)

        round_latency_ms = float(median([r[1] for r in rounds])) if rounds else None
        real_rounds = sum(1 for r in rounds if r[4] == "block")
        probe_rounds = len(rounds) - real_rounds
        committed = sum(r[2] for r in rounds if r[4] == "block")
        throughput_tps = float(committed / self.window_seconds)
        request_latency_ms = float(median([r[1] for r in requests])) if requests else None

        risk_fraction = None
        if fraud_risk_score_pct is not None:
            try:
                risk_fraction = max(0.0, min(1.0, float(fraud_risk_score_pct) / 100.0))
            except (TypeError, ValueError):
                risk_fraction = None

        window_label = int(self.window_seconds)
        features = {
            "node_count": int(active_node_count),
            "network_latency_ms": round_latency_ms,
            "tx_throughput_tps": throughput_tps,
            "security_risk_score": risk_fraction,
            "fault_tolerance_requirement": float(fault_tolerance_requirement),
        }
        sources = {
            "node_count": "Measured: nodes registered in the in-process network",
            "network_latency_ms": (
                f"Measured: median of {len(rounds)} consensus round(s) in the last {window_label}s "
                f"({real_rounds} committed block(s), {probe_rounds} dry-run probe(s) of the active protocol); "
                "in-process timing, NOT peer-to-peer RTT"
                if round_latency_ms is not None else
                f"Unavailable: no block has been forged in the last {window_label}s. "
                "Forge a block so a consensus round can be timed."
            ),
            "tx_throughput_tps": (
                f"Measured: {committed} committed transaction(s) / {window_label}s "
                "(application commit rate, NOT distributed-network throughput)"
            ),
            "security_risk_score": (
                f"Calculated: GNN fraud risk {fraud_risk_score_pct}% converted to {risk_fraction:.4f}"
                if risk_fraction is not None else
                "Unavailable: no GNN fraud-risk score for this transaction"
            ),
            "fault_tolerance_requirement": (fault_tolerance_source or "Configured policy value") + "; not measured",
        }
        result = {
            "features": features,
            "input_sources": sources,
            "source_types": dict(FEATURE_SOURCE_TYPES),
            "measurements": {
                "active_nodes": int(active_node_count),
                "consensus_round_latency_ms": round_latency_ms,
                "consensus_rounds_in_window": len(rounds),
                "committed_rounds_in_window": real_rounds,
                "probe_rounds_in_window": probe_rounds,
                "committed_transactions_in_window": int(committed),
                "window_seconds": self.window_seconds,
                "application_commit_throughput_tps": throughput_tps,
                "request_processing_latency_ms": request_latency_ms,
                "request_processing_note": (
                    "Full HTTP request time (PDF hash, PQC signing, GNN + SHAP). "
                    "Diagnostic only; not used as a selector input."
                ),
                "fraud_risk_score_pct": fraud_risk_score_pct,
                "fault_tolerance_requirement": float(fault_tolerance_requirement),
                "active_consensus_at_measurement": active_consensus,
            },
            "limitations": [
                "Nodes are simulated in one Python process, so latency is an in-process consensus round time, not peer-to-peer RTT.",
                "Throughput is the application's commit rate. A single-user demo commits far fewer than the 1+ TPS seen in the simulated training data.",
                "Energy consumption is not measured because no hardware power telemetry is connected.",
                "The historical training CSV was produced by a simulator; its scales differ from this app's measurements. Out-of-range inputs are therefore expected until the selector is retrained on benchmark-labelled runtime data.",
            ],
        }
        if append_observation:
            self._append_observation(result)
        return result

    # ------------------------------------------------------------ persistence
    def append_observation(self, snapshot: Dict[str, Any], out_of_range_features: Optional[List[str]] = None) -> None:
        self._append_observation(snapshot, out_of_range_features)

    def _append_observation(self, snapshot: Dict[str, Any], out_of_range_features: Optional[List[str]] = None) -> None:
        """Append one unlabelled runtime snapshot. Never invents a consensus label."""
        try:
            self.observations_path.parent.mkdir(parents=True, exist_ok=True)
            features = snapshot.get("features", {})
            m = snapshot.get("measurements", {})
            row = {
                "timestamp_epoch": time(),
                "node_count": features.get("node_count"),
                "network_latency_ms": features.get("network_latency_ms"),
                "tx_throughput_tps": features.get("tx_throughput_tps"),
                "security_risk_score": features.get("security_risk_score"),
                "fault_tolerance_requirement": features.get("fault_tolerance_requirement"),
                "consensus_rounds_in_window": m.get("consensus_rounds_in_window"),
                "committed_transactions_in_window": m.get("committed_transactions_in_window"),
                "window_seconds": m.get("window_seconds"),
                "request_processing_latency_ms": m.get("request_processing_latency_ms"),
                "active_consensus_at_measurement": m.get("active_consensus_at_measurement"),
                "out_of_training_range_features": ";".join(out_of_range_features or []),
                "input_sources_json": json.dumps(snapshot.get("input_sources", {}), ensure_ascii=False),
                "consensus": "",
            }
            with self._observation_lock:
                self._rotate_if_old_schema()
                write_header = (
                    not self.observations_path.exists()
                    or self.observations_path.stat().st_size == 0
                )
                with self.observations_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
        except Exception:
            logger.exception("Could not append runtime metrics observation")

    def _rotate_if_old_schema(self) -> None:
        """If the CSV was written with the old 11-column header, keep it as *_legacy.csv."""
        path = self.observations_path
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header != OBSERVATION_COLUMNS:
            legacy = path.with_name(path.stem + "_legacy" + path.suffix)
            counter = 1
            while legacy.exists():
                legacy = path.with_name(f"{path.stem}_legacy{counter}{path.suffix}")
                counter += 1
            path.rename(legacy)
            logger.warning("Old runtime-metrics schema moved to %s", legacy)


runtime_metrics = RuntimeMetrics()