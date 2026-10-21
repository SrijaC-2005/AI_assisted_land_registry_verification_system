"""Benchmark the app's own consensus implementations and write labelled scenarios.

What it does
------------
1. Builds isolated in-memory networks with app.py's real Network class
   (the real ledger file is never read or written).
2. For each protocol, node count and load level, it forges real blocks with
   realistic transaction payloads and measures round latency exactly the way
   the live app does (median in-process round time). Load level k = blocks
   committed in the app's 300 s window, so throughput = k / 300, the same
   definition the live app uses. Load 0 uses dry-run probes, as the app does.
3. Crosses those measurements with security-risk and fault-tolerance values,
   and labels each scenario with consensus_policy.choose_label():
   the policy-eligible protocol with the lowest measured latency, ties broken
   by fewer messages per round. Every row stores the label reason.

Output
------
  Dataset/Runtime_Consensus_Benchmarks.csv   (training data, with label_reason)
  Dataset/Runtime_Consensus_Measurements.csv (raw per-protocol timings)

Then train:  python train_consensus_selector.py --source benchmark

Honest scope: timings are in-process (nodes are simulated in one Python
process). Labels follow the written policy plus these measurements; they show
which protocol satisfies the policy at lowest measured cost, not that it is
optimal on a real distributed network.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ["LAND_REGISTRY_SKIP_BOOT"] = "1"
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd  # noqa: E402

import app as registry  # noqa: E402  (boot skipped: no ledger I/O)
from consensus_policy import PROTOCOLS, FT_STANDARD, FT_STRICT, choose_label, bft_required  # noqa: E402
from network_metrics import RuntimeMetrics  # noqa: E402

WINDOW_SECONDS = RuntimeMetrics().window_seconds
OUT_SCENARIOS = BASE_DIR / "Dataset" / "Runtime_Consensus_Benchmarks.csv"
OUT_MEASUREMENTS = BASE_DIR / "Dataset" / "Runtime_Consensus_Measurements.csv"
OUT_INDEPENDENT = BASE_DIR / "Dataset" / "Runtime_Consensus_Benchmarks_Independent.csv"
OUT_INDEPENDENT_MEAS = BASE_DIR / "Dataset" / "Runtime_Consensus_Measurements_Independent.csv"


def realistic_tx(i: int) -> dict:
    """Same shape and size as a real queued transfer (PQC signature ~4.4 KB, key ~2.6 KB)."""
    rnd = random.Random(i)
    return {
        "land_id": f"BENCH-{i}",
        "old_owner": "bench_seller",
        "new_owner": "bench_buyer",
        "land_price": rnd.randint(200_000, 20_000_000),
        "document_hash": "%064x" % rnd.getrandbits(256),
        "pqc_signature": "".join(rnd.choices("ABCDEFabcdef0123456789+/", k=4412)),
        "pqc_public_key": "".join(rnd.choices("ABCDEFabcdef0123456789+/", k=2620)),
        "timestamp": "2026-01-01T00:00:00",
        "id": f"bench-{i}",
        "risk_assessment": {"risk_score": 5.0, "risk_level": "LOW"},
    }


def build_network(n: int, protocol: str):
    registry.MAX_NODES = max(registry.MAX_NODES, n)
    net = registry.Network()
    for j in range(1, n):
        net.add_node(f"Node-{j}", stake=100 + 50 * j)
    net.consensus_mode = protocol
    return net


def measure(protocol: str, n: int, k: int, reps: int, tx_counter: list) -> dict:
    latencies, failures = [], 0
    for _ in range(reps):
        net = build_network(n, protocol)
        if k == 0:
            for _ in range(5):
                ms = net.probe_consensus_round()
                if ms is None:
                    failures += 1
                else:
                    latencies.append(ms)
            continue
        for _ in range(k):
            tx_counter[0] += 1
            net.pending_transactions.append(realistic_tx(tx_counter[0]))
            result = net.run_consensus()
            if result.get("success"):
                latencies.append(float(result["round_ms"]))
            else:
                failures += 1
    return {
        "protocol": protocol,
        "node_count": n,
        "load_blocks_per_window": k,
        "tx_throughput_tps": k / WINDOW_SECONDS,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "samples": len(latencies),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", default="4,5,6,8,10")
    parser.add_argument("--loads", default="0,1,2,3,5,10,20,50,100",
                        help="blocks committed per window (throughput = load / window)")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--risks", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--fault-tolerance", default=None,
                        help="comma list; default = the two policy values")
    parser.add_argument("--independent", action="store_true",
                        help="fresh run with node counts, loads, risks and fault-tolerance values NOT in the "
                             "training run, a different seed, written to *_Independent.csv for evaluation only")
    args = parser.parse_args()
    if args.independent:
        args.nodes, args.loads = "7,9", "4,15,30,75"
        args.risks = "0.03,0.12,0.22,0.28,0.32,0.35,0.45,0.65,0.85"
        args.fault_tolerance = args.fault_tolerance or "0.55,0.6,0.7"
    out_scenarios = OUT_INDEPENDENT if args.independent else OUT_SCENARIOS
    out_measurements = OUT_INDEPENDENT_MEAS if args.independent else OUT_MEASUREMENTS

    nodes = [int(x) for x in args.nodes.split(",")]
    loads = [int(x) for x in args.loads.split(",")]
    risks = [float(x) for x in args.risks.split(",")]
    ft_values = ([float(x) for x in args.fault_tolerance.split(",")] if args.fault_tolerance
                 else sorted({FT_STANDARD, FT_STRICT}))

    random.seed(7 if args.independent else 42)
    counter = [0]
    rows = []
    started = time.time()
    total = len(PROTOCOLS) * len(nodes) * len(loads)
    for i, (protocol, n, k) in enumerate(((p, n, k) for p in PROTOCOLS for n in nodes for k in loads), 1):
        m = measure(protocol, n, k, args.reps, counter)
        rows.append(m)
        print(f"[{i:3d}/{total}] {protocol:8s} n={n:<2d} load={k:<3d} "
              f"latency={m['median_latency_ms'] if m['median_latency_ms'] is None else round(m['median_latency_ms'], 3)} ms "
              f"(samples {m['samples']}, failures {m['failures']})")
    measurements = pd.DataFrame(rows)
    OUT_MEASUREMENTS.parent.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(out_measurements, index=False)

    scenarios = []
    for (n, k), group in measurements.dropna(subset=["median_latency_ms"]).groupby(["node_count", "load_blocks_per_window"]):
        latency_by_protocol = dict(zip(group["protocol"], group["median_latency_ms"]))
        tps = k / WINDOW_SECONDS
        for risk in risks:
            for ft in ft_values:
                label, reason = choose_label(n, risk, ft, latency_by_protocol)
                if label is None:
                    continue
                # The live app measures latency under whichever protocol is active,
                # so each scenario is recorded once per possible active protocol.
                for active, latency in latency_by_protocol.items():
                    scenarios.append({
                        "node_count": n,
                        "network_latency_ms": latency,
                        "tx_throughput_tps": tps,
                        "security_risk_score": risk,
                        "fault_tolerance_requirement": ft,
                        "active_protocol_during_measurement": active,
                        "bft_required": bft_required(risk, ft),
                        "consensus": label,
                        "label_reason": reason,
                    })
    df = pd.DataFrame(scenarios)
    df.to_csv(out_scenarios, index=False)

    print(f"\nMeasured in {time.time() - started:.0f}s. Raw timings: {out_measurements}")
    print(f"Labelled scenarios: {len(df)} -> {out_scenarios}")
    print(df["consensus"].value_counts().to_string())
    if args.independent:
        print("\nUsed for evaluation only. Next: python train_consensus_selector.py --source benchmark")
    else:
        print("\nNext: python benchmark_consensus.py --independent   (held-out test scenarios)")


if __name__ == "__main__":
    main()