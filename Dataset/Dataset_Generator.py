# Blockchain Traffic Simulation — Local VS Code Version
# Generates realistic synthetic real-time traffic + blockchain metrics
# Saves CSV locally and prints preview

import math
import random
import os
import datetime
import pandas as pd
import numpy as np

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ---------------- CONFIG ----------------
consensus_types = ['PoW', 'PoS', 'Raft', 'PBFT', 'HotStuff']
minutes = 720  # 12 hours simulation

# Save location (local folder)
OUTPUT_DIR = "output"
OUTPUT_FILE = "Current_Network_Metrics.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

start_time = datetime.datetime.now()
time_index = [start_time + datetime.timedelta(minutes=i) for i in range(minutes)]

# Baseline behavior parameters
params = {
    'PoW':     {'latency': (300,100), 'throughput': (20,80),   'energy': (2.5e6,5e5), 'sec_base': 0.08},
    'PoS':     {'latency': (120,40),  'throughput': (200,500), 'energy': (1.0e5,2.0e4), 'sec_base': 0.12},
    'Raft':    {'latency': (50,20),   'throughput': (300,900), 'energy': (8.0e4,1.5e4), 'sec_base': 0.10},
    'PBFT':    {'latency': (150,60),  'throughput': (150,450), 'energy': (2.0e5,3.0e4), 'sec_base': 0.14},
    'HotStuff':{'latency': (60,30),   'throughput': (400,1500),'energy': (1.2e5,2.5e4), 'sec_base': 0.09},
}

# Initial node counts
node_counts = {c: random.randint(4,12) for c in consensus_types}

# ---------------- HELPERS ----------------

def step_node_count(curr):
    if random.random() < 0.02:
        change = random.choice([-2,-1,1,2])
        curr = max(3, curr + change)
    curr = max(3, curr + random.choice([0,0,0,1,-1]))
    return curr

def vehicle_base(t_idx, minutes_total):
    hour = (start_time + datetime.timedelta(minutes=t_idx)).hour % 24
    morning = math.exp(-((hour-8)**2)/(2*2.5**2))
    evening = math.exp(-((hour-18)**2)/(2*2.5**2))
    base = 200 * (0.6*morning + 0.6*evening) + 100
    trend = 20 * math.sin(2*math.pi*(t_idx/minutes_total))
    return max(20, base + trend)

# ---------------- SIMULATION ----------------

rows = []

for i, t in enumerate(time_index):
    vehicle_count = int(np.round(vehicle_base(i, minutes) + np.random.normal(0, 30)))
    vehicle_variability = max(1.0, abs(np.random.normal(0, 15)))

    for c in consensus_types:
        node_counts[c] = step_node_count(node_counts[c])
        ncount = node_counts[c]
        p = params[c]

        base_latency = np.random.normal(p['latency'][0], p['latency'][1])
        latency_scaling = 1.0 + (ncount - 7)/20.0 + (vehicle_count/1000.0)
        network_latency_ms = max(5, base_latency * latency_scaling + np.random.normal(0,10))

        base_tput = np.random.normal(p['throughput'][0], p['throughput'][1])
        tput_scaling = 1.0 + (ncount - 7)/15.0 - (network_latency_ms/1000.0)
        tx_throughput_tps = max(1, base_tput * tput_scaling * (1.0 + np.random.normal(0,0.05)))

        base_energy = np.random.normal(p['energy'][0], p['energy'][1])
        energy = max(1.0, base_energy * (1 + (ncount-7)/30.0) *
                     (tx_throughput_tps/300.0) * (1+np.random.normal(0,0.08)))

        sec = p['sec_base'] + 0.4*(vehicle_variability/50.0) + np.random.normal(0,0.02)
        if c == 'PoW': sec *= 0.9
        if c == 'HotStuff': sec *= 0.95
        sec = min(0.99, max(0.01, sec))

        base_quorum = {'PoW':0.33, 'PoS':0.34, 'Raft':0.5, 'PBFT':0.66, 'HotStuff':0.67}[c]
        fault_req = min(0.99, base_quorum + 0.2*(vehicle_count/1000.0) +
                        0.3*sec + np.random.normal(0,0.02))

        notes = ''
        if random.random() < 0.01:
            notes = random.choice(['node_join','node_leave','network_partition',
                                   'congestion_event','attack_detected'])

        rows.append({
            'timestamp': t.isoformat(),
            'consensus': c,
            'node_count': ncount,
            'network_latency_ms': round(float(network_latency_ms),2),
            'tx_throughput_tps': round(float(tx_throughput_tps),2),
            'energy_joules_per_min': round(float(energy),2),
            'security_risk_score': round(float(sec),4),
            'vehicle_count': int(vehicle_count),
            'vehicle_count_variability': round(float(vehicle_variability),2),
            'fault_tolerance_requirement': round(float(fault_req),4),
            'notes': notes
        })

df = pd.DataFrame(rows)
df['row_id'] = df.index + 1

cols = ['row_id','timestamp','consensus','node_count','network_latency_ms',
        'tx_throughput_tps','energy_joules_per_min','security_risk_score',
        'vehicle_count','vehicle_count_variability','fault_tolerance_requirement','notes']
df = df[cols]

# Save CSV locally
df.to_csv(out_path, index=False)

# Preview
print("\nSample Preview:\n")
print(df.sample(20, random_state=1))
print(f"\nCSV saved to: {out_path}")
print(f"Total rows generated: {len(df)}")
