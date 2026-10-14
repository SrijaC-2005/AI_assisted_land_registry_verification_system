from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
import time, json, hashlib, random, copy
from datetime import datetime
from risk_assessment.predict import assess_transaction
from risk_assessment.model import LandFraudGNN
from consensus_explainer import explain_consensus
from typing import List, Dict, Any
import os, uuid, itertools
import csv
from numpy import block
from numpy import block
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib
import base64

try:
    from real_dataset_graph import RealDatasetGraph
except Exception:
    RealDatasetGraph = None

from quantcrypt import dss


# Select NIST PQC signature algorithm
algo = dss.MLDSA_65()

# Generate PQC keypair
PQC_PUBLIC_KEY, PQC_PRIVATE_KEY = algo.keygen()


# --- Global Configuration and File Paths (Simplified for execution) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(BASE_DIR, "User_Credential.csv")
MODEL_PATH = os.path.join(BASE_DIR, "FL_SGD", "federated_sgd_global_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "FL_SGD", "fed_label_encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "FL_SGD", "fed_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "Current_Network_Metrics.csv")
CONSENSUS_SELECTOR_DIR = os.path.join(BASE_DIR, "FL_SGD", "consensus_selector")
CONSENSUS_SELECTOR_MODEL_PATH = os.path.join(CONSENSUS_SELECTOR_DIR, "model.pkl")
CONSENSUS_SELECTOR_SCALER_PATH = os.path.join(CONSENSUS_SELECTOR_DIR, "scaler.pkl")
CONSENSUS_SELECTOR_ENCODER_PATH = os.path.join(CONSENSUS_SELECTOR_DIR, "label_encoder.pkl")

NETWORK_STATE_FILE = os.path.join(BASE_DIR, "Blockchain_Ledger.json") # New file for state persistence
USERS: Dict[str, str] = {} # {username: hashed_password}
MAX_NODES = 5

# Interactive graph service. This is separate from
# risk_assessment/graph_builder.py, which remains the GNN graph builder.
# Prefer the mixed HM Land Registry + augmented-owner dataset when supplied.
# This archive does not include that JSON, so fall back to the included CSV.
MIXED_GRAPH_DATASET_PATH = os.path.join(BASE_DIR, "real_augmented_transactions.json")
GRAPH_DATASET_PATH = (
    MIXED_GRAPH_DATASET_PATH
    if os.path.exists(MIXED_GRAPH_DATASET_PATH)
    else os.path.join(BASE_DIR, "Dataset", "land_transactions.csv")
)
GRAPH_NODES_FILE = os.path.join(BASE_DIR, "Graph_Nodes.json")
real_graph = RealDatasetGraph(GRAPH_DATASET_PATH, GRAPH_NODES_FILE) if RealDatasetGraph else None


# --- Credential Management Functions (Unchanged) ---
def load_users_from_csv():
    """Loads users from the CSV file, hashing any plaintext passwords found."""
    global USERS
    USERS = {}
    
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"⚠️ Credential file '{CREDENTIALS_FILE}' not found. Starting fresh.")
        return

    try:
        with open(CREDENTIALS_FILE, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                return

            if header == ['Username', 'HashedPassword']:
                for row in reader:
                    if len(row) == 2:
                        USERS[row[0]] = row[1]
                print(f"✅ Loaded {len(USERS)} users from secure CSV format.")
            elif header == ['Username', 'Password']:
                print("⚠️ Found plaintext passwords. Hashing and rewriting file securely...")
                for row in reader:
                    if len(row) == 2:
                        username, password = row
                        USERS[username] = generate_password_hash(password)
                save_users_to_csv()
                print(f"✅ Hashed and loaded {len(USERS)} users.")
            else:
                print("❌ CSV header format is incorrect. Skipping user load.")

    except Exception as e:
        print(f"❌ Error loading credentials: {e}")

def save_users_to_csv():
    """Writes the current USERS dictionary (hashed passwords) back to the CSV."""
    try:
        with open(CREDENTIALS_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Username', 'HashedPassword'])
            for username, hashed_password in USERS.items():
                writer.writerow([username, hashed_password])
        print(f"💾 Saved {len(USERS)} users to CSV.")
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")

# --- ML Model and Data Setup (Unchanged in logic) ---
try:
    import pandas as pd
    import numpy as np
    import joblib
    
    model_ready = False
    model = None
    label_encoder = None
    df_traffic = None
    scaler = None

    ML_EXPECTED_FEATURES = [
        'node_count', 
        'network_latency_ms', 
        'tx_throughput_tps', 
        'energy_joules_per_min', 
        'security_risk_score', 
        'vehicle_count', 
        'vehicle_count_variability', 
        'fault_tolerance_requirement'
    ]
    MODEL_FEATURES = []

    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(DATA_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            scaler = joblib.load(SCALER_PATH)

            df_traffic = pd.read_csv(DATA_PATH)
            MODEL_FEATURES = ML_EXPECTED_FEATURES

            if not all(feature in df_traffic.columns for feature in ML_EXPECTED_FEATURES):
                print("❌ Missing ML features.")
            else:
                model_ready = True
                print("🧠 Federated SGD model loaded. Dynamic consensus active.")

        except Exception as e:
            print(f"⚠️ Failed to load ML assets: {e}")

    else:
        print("⚠️ ML files not found. Running in static consensus mode.")

except Exception as e:
    print(f"⚠️ ML library imports failed. Error: {e}")
    model_ready = False
    pd = None
    np = None
    joblib = None


# Snapshot of the actual existing dynamic-consensus decision, used only for explanation.
last_consensus_decision = None

# Separate consensus-selector artifacts: never overwrite the existing federated fraud model.
selector_model = selector_scaler = selector_label_encoder = None
selector_ready = False
selector_training_frame = None
try:
    import pandas as pd
    import joblib
    if all(os.path.exists(path) for path in (
        CONSENSUS_SELECTOR_MODEL_PATH, CONSENSUS_SELECTOR_SCALER_PATH,
        CONSENSUS_SELECTOR_ENCODER_PATH, DATA_PATH
    )):
        selector_model = joblib.load(CONSENSUS_SELECTOR_MODEL_PATH)
        selector_scaler = joblib.load(CONSENSUS_SELECTOR_SCALER_PATH)
        selector_label_encoder = joblib.load(CONSENSUS_SELECTOR_ENCODER_PATH)
        selector_training_frame = pd.read_csv(DATA_PATH)
        selector_ready = True
        print("🧠 Trained consensus selector loaded; Module 4 LIME explains this classifier.")
    else:
        print("⚠️ Trained consensus selector not found. Run python train_consensus_selector.py.")
except Exception as selector_load_error:
    print(f"⚠️ Consensus selector unavailable: {selector_load_error}")
    selector_ready = False


app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Initial Load of Users ---
load_users_from_csv()

# -----------------------------
# Data Structures and Utilities (Blockchain Logic)
# -----------------------------
def compute_hash(block_dict):
    """Computes the SHA256 hash of a block dictionary."""
    s = json.dumps(block_dict, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def hash_pdf(file_bytes: bytes) -> str:
    """SHA256 hash of uploaded PDF"""
    return hashlib.sha256(file_bytes).hexdigest()


def pqc_sign(data: str) -> str:
    signature = algo.sign(
        PQC_PRIVATE_KEY,
        data.encode()
    )
    return base64.b64encode(signature).decode()
def get_transaction_history(self):

    history = []

    # Get transactions from blockchain
    for block in self.chain:
        for tx in block.transactions:

            # Ignore genesis block
            if "land_id" in tx:
                history.append(tx)

    # Also include pending transactions
    for tx in self.pending_transactions:
        if "land_id" in tx:
            history.append(tx)

    return history


def pqc_public_key_b64() -> str:
    return base64.b64encode(PQC_PUBLIC_KEY).decode()


class Block:
    """Represents a single block in the blockchain."""
    def __init__(self, index, timestamp, transactions, previous_hash, nonce=0, proposer=None):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.proposer = proposer or 'Genesis'

    def to_dict(self):
        """Returns a dictionary representation of the block."""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'proposer': self.proposer,
        }

    def hash(self):
        """Calculates the hash of the block."""
        return compute_hash(self.to_dict())

class Network:
    """Simulates the multi-node blockchain network with governance."""

    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.blocked_transactions: List[Dict] = []
        self.nodes: Dict[str, Dict] = {}
        self.consensus_mode: str = 'PoW'
        self.difficulty: int = 3
        self.consensus_algos: List[str] = ['PoW', 'PoS', 'Raft', 'PBFT', 'HotStuff']
        self.active_users: Dict[str, bool] = {} # {username: True}
        self.pending_users: Dict[str, List[str]] = {} # {username: [voter1, voter2, ...]}
        self.add_node('Node-A', is_initial=True)
        self.create_genesis_block()

    def check_duplicate_transaction(self, tx):

        land_id = tx["land_id"]
        document_hash = tx["document_hash"]
        seller = tx["old_owner"]
        buyer = tx["new_owner"]

        # -------------------------
        # Check Pending Queue
        # -------------------------
        for pending in self.pending_transactions:

            # Same Document Hash
            if pending.get("document_hash") == document_hash:
                return False, "This document is already used in another transaction."

            # Same complete transaction
            if (
                pending.get("old_owner") == seller
                and pending.get("new_owner") == buyer
                and pending.get("land_id") == land_id
            ):
                return False, "Same transaction already exists in Pending Queue."

        # -------------------------
        # Check Blockchain Ledger
        # -------------------------
        for block in self.chain:

            for ledger_tx in block.transactions:

                # Ignore Genesis Block
                if "land_id" not in ledger_tx:
                    continue

                # Same Document Hash
                if ledger_tx.get("document_hash") == document_hash:
                    return False, "This document is already registered in the Ledger."

                # Same complete transaction
                if (
                    ledger_tx.get("old_owner") == seller
                    and ledger_tx.get("new_owner") == buyer
                    and ledger_tx.get("land_id") == land_id
                ):
                    return False, "Same transaction already exists in Ledger."

        return True, "No duplicates."

    def create_genesis_block(self):
        if not self.chain:
            self.chain.append(Block(
                index=0,
                timestamp=time.time(),
                transactions=[{'message': 'Genesis Block', 'predicted_consensus': 'PoW'}],
                previous_hash="0" * 64
            ))

    def get_last_block(self) -> Block:
        return self.chain[-1]
    
    # --- Other blockchain methods (run_pow, add_transaction, etc.) remain the same ---
    # (Removed for brevity, but they are fully included in the final file below)
    
    def add_node(self, name, stake=100, is_initial=False):
        if len(self.nodes) >= MAX_NODES and not is_initial:
            return False, "Max nodes reached (5)."

        if name in self.nodes:
            return False, f"Node {name} already exists."

        self.nodes[name] = {
            'stake': stake,
            'address': str(uuid.uuid4()),
            'chain_length': len(self.chain)
        }
        if is_initial:
             # Ensure the initial node is the only one if starting fresh
             self.nodes = {name: self.nodes[name]}
        return True, f"Node {name} added with stake {stake}."

    def remove_node(self, name):
        if name in self.nodes:
            del self.nodes[name]
            return True
        return False
    def get_transaction_history(self):
        history = []

        # Get transactions already stored in blockchain
        for block in self.chain:
            for tx in block.transactions:
                if isinstance(tx, dict) and "land_id" in tx:
                    history.append(tx)

        # Also include transactions waiting in pending queue
        for tx in self.pending_transactions:
            if isinstance(tx, dict) and "land_id" in tx:
                history.append(tx)

        return history
    def add_land_transaction(self, old_owner, new_owner, land_price, pdf_bytes, land_id):

        if land_price <= 0:
            return False, "Land price must be positive."

        # 1. Hash document
        document_hash = hash_pdf(pdf_bytes)

        # 2. Sign document hash using PQC
        pqc_sig = pqc_sign(document_hash)

        tx = {
            "land_id": land_id,
            "old_owner": old_owner,
            "new_owner": new_owner,
            "land_price": land_price,
            "document_hash": document_hash,
            "pqc_signature": pqc_sig,
            "pqc_public_key": pqc_public_key_b64(),
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4())
        }

        # ------------------------------------------
        # DUPLICATE CHECK
        # ------------------------------------------

        ok, message = self.check_duplicate_transaction(tx)

        if not ok:
            return False, message

        # ------------------------------------------
        # GNN RISK ASSESSMENT
        # ------------------------------------------

        print("\n>>> REACHED GNN ASSESSMENT <<<")

        historical_transactions = self.get_transaction_history()

        risk_result = assess_transaction(
            tx,
            historical_transactions
        )

        print("\n========== GNN DECISION ==========")
        print("Fraud Probability:", risk_result["fraud_probability"])
        print("Risk Score:", risk_result["risk_score"])
        print("Risk Level:", risk_result["risk_level"])
        print("Recommendation:", risk_result["recommendation"])
        print("==================================\n")
        print("\n========== SHAP EXPLANATION ==========")

        for feature, shap_value in risk_result["shap_explanation"].items():
            print(f"{feature}: {shap_value:+.6f}")

        print("======================================\n")
        # Store risk result
        tx["risk_assessment"] = risk_result

        # ------------------------------------------
        # HIGH-RISK TRANSACTION
        # ------------------------------------------

        if risk_result["risk_level"] == "HIGH":

            blocked_tx = {
                "transaction": tx,
                "risk_assessment": risk_result,
                "blocked_at": datetime.now().isoformat(),
                "reason": "AI detected HIGH risk"
            }

            self.blocked_transactions.append(blocked_tx)

            print("HIGH-RISK TRANSACTION BLOCKED.")
            print("Transaction was NOT added to pending queue.")
            print(
                "Current blocked transactions:",
                len(self.blocked_transactions)
            )

            return False, {
                "blocked": True,
                "transaction": tx,
                "risk_assessment": risk_result,
                "message": "Transaction blocked because AI detected HIGH risk."
            }


        # ------------------------------------------
        # LOW / MEDIUM → EXISTING PENDING QUEUE
        # ------------------------------------------

        self.pending_transactions.append(tx)

        print("Transaction added to pending queue.")
        print(
            "Current pending transactions:",
            len(self.pending_transactions)
        )

        # IMPORTANT:
        # Do NOT call save_network_state() here.
        # The API persistence wrapper will save it.

        return True, tx

    def set_consensus(self, mode):
        if mode in self.consensus_algos:
            self.consensus_mode = mode
            return True
        return False

    def run_pow(self, proposer_node: str) -> Block:
        last_block = self.get_last_block()

        # Take ONLY the first pending transaction
        tx = self.pending_transactions[0]

        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=[tx],
            previous_hash=last_block.hash(),
            proposer=proposer_node
        )

        # Consensus information
        new_block.transactions[0]['predicted_consensus'] = 'PoW'

        while True:
            new_block.nonce += 1
            hash_attempt = new_block.hash()

            if hash_attempt.startswith('0' * self.difficulty):

                # Remove ONLY the transaction that was forged
                self.pending_transactions.pop(0)

                # Add the block
                self.chain.append(new_block)

                return new_block

            if new_block.nonce > 100000:
                new_block.nonce = 0
                new_block.timestamp = time.time()

    def run_pos(self, proposer_node: str) -> Block:
        stakes = [self.nodes[name]['stake'] for name in self.nodes]
        total_stake = sum(stakes)
        if total_stake == 0:
             validator = proposer_node
        else:
             weights = [s / total_stake for s in stakes]
             validator = random.choices(list(self.nodes.keys()), weights=weights, k=1)[0]

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=[self.pending_transactions[0]],
            previous_hash=last_block.hash(),
            proposer=validator
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'PoS'
             
        self.pending_transactions.pop(0)
        self.chain.append(new_block)
        return new_block

    def run_pbft(self, proposer_node: str) -> Block:
        # PBFT requires at least 4 nodes (f = 1)
        n = len(self.nodes)
        if n < 4:
            return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=[self.pending_transactions[0]],
            previous_hash=last_block.hash(),
            proposer=proposer_node
        )

        if new_block.transactions:
            new_block.transactions[0]['predicted_consensus'] = 'PBFT'

        # PBFT theory:
        # n = 3f + 1  =>  f = floor((n - 1) / 3)
        f = (n - 1) // 3
        quorum = 2 * f + 1

        # Assume honest majority (simulation)
        commit_count = n - f - 1  # excluding proposer

        # Debug (optional)
        # print(f"PBFT commit votes: {commit_count}, quorum: {quorum}")

        if commit_count >= quorum:
            self.pending_transactions.pop(0)
            self.chain.append(new_block)
            return new_block
        else:
            return None


    def run_raft(self, proposer_node: str) -> Block:
        leader = random.choice(list(self.nodes.keys()))
        if leader != proposer_node:
             return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=[self.pending_transactions[0]],
            previous_hash=last_block.hash(),
            proposer=leader
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'Raft'

        self.pending_transactions.pop(0)    
        self.chain.append(new_block)
        return new_block

    def run_hotstuff(self, proposer_node: str) -> Block:
        if len(self.nodes) < 3:
             return None

        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=[self.pending_transactions[0]],
            previous_hash=last_block.hash(),
            proposer=proposer_node
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'HotStuff'

        self.pending_transactions.pop(0)
        self.chain.append(new_block)
        return new_block


    def run_consensus(self):
        if not self.pending_transactions:
            return {'success': False, 'result': 'No pending transactions to process.'}

        proposer_node = random.choice(list(self.nodes.keys()))
        new_block = None

        if self.consensus_mode == 'PoW':
            new_block = self.run_pow(proposer_node)
        elif self.consensus_mode == 'PoS':
            new_block = self.run_pos(proposer_node)
        elif self.consensus_mode == 'PBFT':
            new_block = self.run_pbft(proposer_node)
        elif self.consensus_mode == 'Raft':
            new_block = self.run_raft(proposer_node)
        elif self.consensus_mode == 'HotStuff':
            new_block = self.run_hotstuff(proposer_node)

        if new_block:
            for node in self.nodes.values():
                 node['chain_length'] = len(self.chain)

            return {'success': True, 'result': f'Block {new_block.index} forged via {self.consensus_mode}'}
        else:
            return {'success': False, 'result': f'Consensus failed for {self.consensus_mode}.'}


    def resolve_conflicts(self):
        longest_chain_length = max(node['chain_length'] for node in self.nodes.values())
        if len(self.chain) < longest_chain_length:
            return {'message': 'Conflict detected, but resolution logic skipped for simulation.'}
        else:
            return {'message': 'Local chain is the longest. No conflict.'}

# --- State Persistence Functions for Multi-User Support ---

def save_network_state():
    """Safely saves the network state to disk using atomic replacement."""

    global network

    try:
        state = {
            'chain': [
                block.to_dict()
                for block in network.chain
            ],

            'pending_transactions':
                network.pending_transactions,

            'blocked_transactions':
                network.blocked_transactions,

            'nodes':
                network.nodes,

            'consensus_mode':
                network.consensus_mode,

            'difficulty':
                network.difficulty,

            'consensus_algos':
                network.consensus_algos,

            'active_users':
                network.active_users,

            'pending_users':
                network.pending_users,
        }

        # Write to temporary file FIRST
        temp_file = NETWORK_STATE_FILE + ".tmp"

        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(
                state,
                f,
                indent=4
            )

            # Make sure everything is physically written
            f.flush()
            os.fsync(f.fileno())

        # Replace old ledger ONLY after successful JSON writing
        os.replace(
            temp_file,
            NETWORK_STATE_FILE
        )

        print(
            f"💾 State saved | "
            f"Blocks: {len(network.chain)} | "
            f"Pending: {len(network.pending_transactions)}"
        )

    except Exception as e:

        print(
            f"❌ Error saving network state: {e}"
        )

        # Remove failed temporary file if it exists
        temp_file = NETWORK_STATE_FILE + ".tmp"

        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def load_network_state():

    global network

    # ----------------------------------------
    # Existing blockchain file
    # ----------------------------------------

    if os.path.exists(NETWORK_STATE_FILE):

        try:

            with open(
                NETWORK_STATE_FILE,
                'r'
            ) as f:

                state = json.load(f)

            # Create fresh object in memory
            network = Network()

            # Remove automatically created genesis block
            network.chain = []

            # --------------------------------
            # Restore blockchain
            # --------------------------------

            for block_data in state.get(
                'chain',
                []
            ):

                block = Block(
                    index=block_data['index'],
                    timestamp=block_data['timestamp'],
                    transactions=block_data['transactions'],
                    previous_hash=block_data['previous_hash'],
                    nonce=block_data.get(
                        'nonce',
                        0
                    ),
                    proposer=block_data.get(
                        'proposer'
                    )
                )

                network.chain.append(block)

            # --------------------------------
            # Restore pending transactions
            # --------------------------------

            network.pending_transactions = state.get(
                'pending_transactions',
                []
            )
            network.blocked_transactions = state.get(
                'blocked_transactions',
                []
            )
            # --------------------------------
            # Restore nodes
            # --------------------------------

            network.nodes = state.get(
                'nodes',
                network.nodes
            )

            # --------------------------------
            # Restore settings
            # --------------------------------

            network.consensus_mode = state.get(
                'consensus_mode',
                'PoW'
            )

            network.difficulty = state.get(
                'difficulty',
                3
            )

            network.consensus_algos = state.get(
                'consensus_algos',
                [
                    'PoW',
                    'PoS',
                    'Raft',
                    'PBFT',
                    'HotStuff'
                ]
            )

            # --------------------------------
            # Restore users
            # --------------------------------

            network.active_users = state.get(
                'active_users',
                {}
            )

            network.pending_users = state.get(
                'pending_users',
                {}
            )

            # Only create genesis if saved chain
            # was actually empty
            if len(network.chain) == 0:
                network.create_genesis_block()

            print(
                f"✅ Blockchain loaded successfully. "
                f"Blocks: {len(network.chain)}, "
                f"Pending: {len(network.pending_transactions)}"
            )

        except Exception as e:

            print(
                "❌ Error loading blockchain:",
                e
            )

            # Do not save anything here
            print("⚠️ Existing ledger was NOT overwritten.")

    # ----------------------------------------
    # No blockchain file
    # ----------------------------------------

    else:

        print(
            "ℹ️ No existing blockchain ledger found. "
            "Creating new blockchain."
        )

        network = Network()

        save_network_state()

    ensure_default_users()
def online_update(features_scaled, true_label_str):
    global model

    if not model_ready:
        return

    try:
        # encode real label from dataset
        true_label_encoded = label_encoder.transform([true_label_str])[0]

        model.partial_fit(
            features_scaled,
            np.array([true_label_encoded])
        )


        joblib.dump(model, MODEL_PATH)

        print(f"⚡ Learned from real label: {true_label_str}")

    except Exception as e:
        print("Online update error:", e)


def ensure_default_users():
    """Ensures at least two default users exist and are active."""

    global network

    # -----------------------------
    # 1. Ensure default credentials
    # -----------------------------
    default_users = {
        'Sengathir': '12245',
        'UserB': 'defaultpass',
        'vishal': 'vishal@123'
    }

    credentials_changed = False

    for username, password in default_users.items():
        if username not in USERS:
            USERS[username] = generate_password_hash(password)
            credentials_changed = True

    if credentials_changed:
        save_users_to_csv()
        print(f"✅ Initialized {len(default_users)} default users.")

    # -----------------------------
    # 2. Ensure network exists
    # -----------------------------
    if network is None:
        load_network_state()   # initializes `network`

    # -----------------------------
    # 3. Ensure minimum 2 active users (these are founders)
    # -----------------------------
    initial_count = len(network.active_users)
    
    for username in default_users:
        if username in USERS and username not in network.active_users and username not in network.pending_users:
            network.active_users[username] = True
            print(f"✅ Activated default user: {username}")

    # If first time setup, also add them to nodes so they can participate
    for username in default_users:
        if username in USERS and username not in network.active_users and username in network.pending_users:
            # Move from pending to active
            del network.pending_users[username]
            network.active_users[username] = True
            print(f"✅ Promoted pending user to active: {username}")

    # Add these users as nodes if they're not already
    for i, username in enumerate(default_users):
        node_name = f"Node-{username}"
        if username in network.active_users and node_name not in network.nodes:
            network.add_node(node_name, stake=200 + (i * 100), is_initial=(i == 0))
            print(f"✅ Added node for user: {username}")

    # -----------------------------
    # 4. Prune invalid active users
    # -----------------------------
    for user in list(network.active_users.keys()):
        if user not in USERS:
            del network.active_users[user]

    # Remove from pending if already active
    for user in list(network.pending_users.keys()):
        if user in network.active_users:
            del network.pending_users[user]

    # Print status
    if initial_count != len(network.active_users):
        print(f"👥 Active users: {list(network.active_users.keys())}")
        print(f"⏳ Pending users: {list(network.pending_users.keys())}")

    # -----------------------------
    # 5. Persist valid state
    # -----------------------------
    if len(network.active_users) >= 2:
        save_network_state()


# -------------------------------------
# Application Setup
# -------------------------------------

network = None
load_network_state()  # will call ensure_default_users internally

# A simple wrapper to handle persistence for API/Action routes
def persist_action(func):

    def wrapper(*args, **kwargs):

        load_network_state()

        response = func(
            *args,
            **kwargs
        )

        save_network_state()

        return response

    wrapper.__name__ = (
        func.__name__
        + '_persisted'
    )

    return wrapper


@app.before_request
def check_login():
    """Checks if the user is logged in before allowing access to app routes."""
    # Exclude login, signup, and static files
    if request.path in ['/login', '/signup'] or request.path.startswith('/static') or request.path.startswith('/api'):
        return
    if 'logged_in' not in session:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login using CSV-backed credentials and community approval."""
    load_network_state() # Load state for checking active users
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS:
            if check_password_hash(USERS[username], password):
                if username in network.active_users:
                    session['logged_in'] = True
                    session['username'] = username
                    return redirect(url_for('index'))
                else:
                    message = "Login successful, but your account is **PENDING COMMUNITY APPROVAL**. Check back later."
            else:
                message = "Invalid password."
        else:
            message = "Username not found. Please sign up."
    
    save_network_state() # Save state if network loaded correctly (no functional change here)
    return render_template_string(LOGIN_HTML, message=message, is_login=True)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles new user sign up, placing them in the PENDING list."""
    load_network_state() # Load state for checking existing users
    message = ""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            message = "Username and password are required."
        elif username in USERS:
            message = f"Username '{username}' already exists."
        else:
            # 1. Hash the password and save the new user credential
            hashed_password = generate_password_hash(password)
            USERS[username] = hashed_password
            save_users_to_csv()
            
            # 2. Add user to pending list (initializes with an empty voter list)
            network.pending_users[username] = []
            save_network_state() # Save network state with new pending user
            
            message = f"Sign up successful! Your account, '{username}', is pending **COMMUNITY APPROVAL**. Please check back later."
            return render_template_string(LOGIN_HTML, message=message, is_login=True)

    return render_template_string(LOGIN_HTML, message=message, is_login=False)


@app.route('/logout')
def logout():
    """Logs the user out."""
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    """The main application page."""
    load_network_state() # Load state before rendering
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Allow both active and pending users to access the interface
    username = session.get('username')
    if username not in network.active_users and username not in network.pending_users and username not in USERS:
        return redirect(url_for('login'))
    
    return render_template_string(INDEX_HTML)

def _all_transactions_flat():
    """Flatten ledger + pending transactions for graph construction."""
    txs = []
    for block in network.chain:
        for tx in block.transactions:
            if isinstance(tx, dict) and "land_id" in tx:
                item = dict(tx)
                item["source"] = "ledger"
                item["block_index"] = block.index
                txs.append(item)

    for tx in network.pending_transactions:
        if isinstance(tx, dict) and "land_id" in tx:
            item = dict(tx)
            item["source"] = "pending"
            item["block_index"] = None
            txs.append(item)

    return txs


def _all_graph_transactions_flat():
    """Flatten ledger, pending and blocked transactions for graph investigation.

    Blocked transactions are included only in the visualization/search layer.
    They are never inserted into the blockchain chain by this helper.
    """
    txs = _all_transactions_flat()
    for blocked_item in network.blocked_transactions:
        tx = blocked_item.get("transaction") if isinstance(blocked_item, dict) else None
        if not isinstance(tx, dict) or "land_id" not in tx:
            continue
        item = dict(tx)
        item["source"] = "blocked"
        item["blocked"] = True
        item["block_index"] = None
        txs.append(item)
    return txs


@app.route('/api/graph/real', methods=['GET'])
def real_dataset_graph_route():
    """Return interactive graph data from the mixed UK HM Land Registry + synthetic-owner dataset and live blockchain state."""
    if real_graph is None:
        return jsonify({
            "success": False,
            "message": "Graph service is unavailable because real_dataset_graph.py could not be loaded."
        }), 500

    load_network_state()
    try:
        limit = min(max(int(request.args.get("limit", 600)), 1), 1500)
    except (TypeError, ValueError):
        limit = 600

    postcode = request.args.get("postcode", "").strip()
    property_id = request.args.get("property_id", "").strip()
    include_blockchain = request.args.get("include_blockchain", "1") != "0"
    recent_only = request.args.get("recent_only", "0") != "0"
    focus_type = request.args.get("focus_type", "").strip() or None
    focus_label = request.args.get("focus_label", "").strip() or None
    admin_only = request.args.get("admin_only", "0") == "1"

    graph = real_graph.build_graph(
        limit=limit,
        postcode=postcode or None,
        property_id=property_id or None,
        recent_only=recent_only and not focus_label,
        focus_type=focus_type,
        focus_label=focus_label,
        admin_only=admin_only,
        blockchain_transactions=_all_graph_transactions_flat() if include_blockchain else [],
        blockchain_nodes=network.nodes if include_blockchain else {},
    )
    return jsonify({"success": True, **graph})


@app.route('/api/graph/add_node', methods=['POST'])
@persist_action
def add_graph_node():
    if real_graph is None:
        return jsonify({"success": False, "message": "Graph service unavailable."}), 500

    data = request.json or {}
    node_type = str(data.get("type", "Custom")).strip() or "Custom"
    label = str(data.get("label", "")).strip()
    if not label:
        return jsonify({"success": False, "message": "Graph node label is required."}), 400

    node = real_graph.add_custom_node(node_type, label, data.get("metadata") or {})
    return jsonify({
        "success": True,
        "message": f"Graph node '{label}' added.",
        "node": node,
    })


@app.route('/api/graph/add_edge', methods=['POST'])
@persist_action
def add_graph_edge():
    if real_graph is None:
        return jsonify({"success": False, "message": "Graph service unavailable."}), 500

    data = request.json or {}
    source = str(data.get("source", "")).strip()
    target = str(data.get("target", "")).strip()
    relation = str(data.get("relation", "RELATED_TO")).strip() or "RELATED_TO"

    if not source or not target:
        return jsonify({
            "success": False,
            "message": "Source and target node IDs are required."
        }), 400

    edge = real_graph.add_custom_edge(source, target, relation)
    if not edge:
        return jsonify({
            "success": False,
            "message": "Source/target node must exist in the custom graph."
        }), 400

    return jsonify({"success": True, "message": "Graph edge added.", "edge": edge})


# --- Simulation API Routes ---
@app.route('/api/state')
def get_state():
    load_network_state()
    last_block = network.get_last_block()
    
    # Get prediction from the latest transaction in the last block 
    # or use a default if the chain is just starting.
    prediction = "PoW" # Default
    trigger_source = "System (Default)"
    
    if last_block and last_block.transactions:
        # Assuming the 'predicted_consensus' was stored in the transaction data
        prediction = last_block.transactions[0].get('predicted_consensus', "PoW")
        trigger_source = "ML Model (Network Load)"

    return jsonify({
        'chain': [block.to_dict() for block in network.chain],
        'pending_transactions': network.pending_transactions,
        'blocked_transactions': network.blocked_transactions,
        'nodes': network.nodes,
        'consensus_mode': network.consensus_mode,
        'predicted_mode': prediction,       # <--- ADD THIS
        'prediction_trigger': trigger_source, # <--- ADD THIS
        'difficulty': network.difficulty,
        'consensus_algos': network.consensus_algos,
        'ml_ready': model_ready,
        'last_block_hash': last_block.hash(),
        'node_count': len(network.nodes),
        'current_user': session.get('username', 'Guest')
    })

# --- New Governance API Routes ---

@app.route('/api/get_users')
def get_users():
    load_network_state()
    active_count = len(network.active_users)
    majority_threshold = int(active_count / 2) + 1
    
    # Prepare pending list for UI
    pending_list = []
    for user, voters in network.pending_users.items():
        pending_list.append({
            'username': user,
            'voters': voters,
            'vote_count': len(voters),
            'voted_by_current': session.get('username') in voters
        })

    return jsonify({
        'active_users': list(network.active_users.keys()),
        'pending_users': pending_list,
        'active_count': active_count,
        'majority_threshold': majority_threshold,
    })

@app.route('/api/vote_user', methods=['POST'])
@persist_action
def vote_user():
    voter = session.get('username')
    data = request.json
    target_user = data.get('username')

    if target_user not in network.pending_users:
        return jsonify({'success': False, 'message': 'User not found in pending list.'})

    if voter not in network.active_users:
        return jsonify({'success': False, 'message': 'Only active users can vote.'})

    current_voters = network.pending_users[target_user]

    if voter in current_voters:
        return jsonify({'success': False, 'message': 'You already voted.'})

    current_voters.append(voter)

    active_count = len(network.active_users)
    majority = (active_count // 2) + 1

    if len(current_voters) >= majority:
        network.active_users[target_user] = True
        del network.pending_users[target_user]
        return jsonify({
            'success': True,
            'approved': True,
            'message': f"User '{target_user}' approved by majority ({len(current_voters)}/{active_count})"
        })

    return jsonify({
        'success': True,
        'approved': False,
        'message': f"Vote recorded ({len(current_voters)}/{majority})"
    })


# --- Persisted Simulation API Routes ---

@app.route('/api/add_node', methods=['POST'])
@persist_action
def add_node():
    data = request.json
    name = data.get('name')
    stake = int(data.get('stake', 100))
    if not name:
        name = 'Node-' + ''.join(random.choices('BCDEF', k=1)) + str(len(network.nodes) + 1)
    success, message = network.add_node(name, stake)
    return jsonify({'success': success, 'message': message})

@app.route('/api/remove_node', methods=['POST'])
@persist_action
def remove_node():
    data = request.json
    name = data.get('name')
    if name not in network.nodes:
        return jsonify({'success': False, 'message': f"Node {name} not found."})
    if len(network.nodes) <= 1:
        return jsonify({'success': False, 'message': "Cannot remove the last node."})
    success = network.remove_node(name)
    return jsonify({'success': success, 'message': f"Node {name} removed."})

@app.route('/api/add_land_tx', methods=['POST'])

def add_land_tx():
    current_user = session.get('username', 'Anonymous')
    
    # Check if user is active (approved by community)
    
    if current_user not in network.active_users:
        return jsonify({'success': False, 'message': f'User "{current_user}" is not yet approved by the community. Please wait for approval.'}), 403
    
    old_owner = current_user
    land_id = request.form.get('land_id')
    new_owner = request.form.get('new_owner')
    land_price = int(request.form.get('land_price', 0))
    pdf_file = request.files.get('pdf')

    if not pdf_file:
        return jsonify({'success': False, 'message': 'PDF document required.'})

    pdf_bytes = pdf_file.read()

    success, result = network.add_land_transaction(
        old_owner=old_owner,
        new_owner=new_owner,
        land_price=land_price,
        pdf_bytes=pdf_bytes,
        land_id=land_id
    )

    if not success:

        # HIGH-risk transaction was blocked by AI
        if isinstance(result, dict) and result.get("blocked"):

            print("Saving HIGH-risk blocked transaction...")
            save_network_state()

            return jsonify({
                'success': False,
                'blocked': True,
                'message': result["message"],
                'transaction': result["transaction"],
                'risk_assessment': result["risk_assessment"]
            })

        # Other failures such as duplicate transaction
        return jsonify({
            'success': False,
            'message': result
        })


    print("Saving transaction to blockchain state...")
    save_network_state()

    print(
        "Saved pending transactions:",
        len(network.pending_transactions)
    )

    print(
        "Saved pending transactions:",
        len(network.pending_transactions)
    )

    # Preserve the project's existing dynamic-consensus pipeline: use the same
    # labelled network-metrics row, model prediction, online update, and set_consensus.
    # Module 4 records and explains this exact decision; it does not make a second one.
    global last_consensus_decision
    predicted_consensus = network.consensus_mode
    decision_features = None
    decision_model_snapshot = None
    true_label = None
    try:
        if model_ready and df_traffic is not None and not df_traffic.empty:
            sampled_row = df_traffic.sample(n=1).iloc[0]
            decision_features = {feature: float(sampled_row[feature]) for feature in MODEL_FEATURES}
            features_df = sampled_row[MODEL_FEATURES].to_frame().T
            scaled_features = scaler.transform(features_df.values)
            # Snapshot the exact model state that made this prediction, before online update.
            decision_model_snapshot = copy.deepcopy(model)
            prediction = model.predict(scaled_features)[0]
            predicted_consensus = str(label_encoder.inverse_transform([int(prediction)])[0])
            if 'consensus' in sampled_row.index:
                true_label = str(sampled_row['consensus'])
            network.set_consensus(predicted_consensus)
            if true_label is not None:
                online_update(scaled_features, true_label)
            print(f"Dynamic consensus selected: {predicted_consensus}; active mode: {network.consensus_mode}; dataset label: {true_label}")
            last_consensus_decision = {
                'selected_consensus': predicted_consensus,
                'active_consensus': network.consensus_mode,
                'features': decision_features,
                'model_snapshot': decision_model_snapshot,
                'source': 'Existing dynamic consensus pipeline using a sampled row from Current_Network_Metrics.csv',
                'true_label': true_label,
                'transaction_id': result.get('transaction_id', result.get('id')),
                'timestamp': time.time(),
            }
        else:
            last_consensus_decision = None
            print('Dynamic consensus model unavailable; existing active consensus retained.')
    except Exception as consensus_error:
        app.logger.exception('Existing dynamic consensus selection failed')
        # Do not claim a new selection if the existing pipeline failed.
        last_consensus_decision = None

    result['selected_consensus'] = predicted_consensus
    result['executed_consensus'] = network.consensus_mode

    return jsonify({
        'success': True,
        'message': 'Land transaction added to queue.',
        'transaction': result
    })


@app.route('/api/consensus/explanation', methods=['GET'])
def consensus_explanation_route():
    """Explain the actual decision made by the existing dynamic consensus pipeline."""
    try:
        load_network_state()
        decision = last_consensus_decision
        active_mode = getattr(network, 'consensus_mode', 'PoW')
        if not decision:
            return jsonify({
                'module': 'Module 4 — Explainable Dynamic Consensus',
                'error': 'No dynamic-consensus decision has been captured in this server session yet. Submit a land transaction to run the existing selector, then refresh this explanation.',
                'selected_consensus': None,
                'current_consensus': active_mode,
                'lime_explanation': {'method': 'LIME not run', 'features': [], 'warning': 'No actual selection event is available to explain.'},
                'decision_method': 'Existing dynamic consensus pipeline',
            })
        if not model_ready or scaler is None or label_encoder is None or df_traffic is None:
            return jsonify({'error': 'The existing dynamic consensus model or its dataset is unavailable; cannot explain the actual selection.'}), 503

        # Use the exact feature row and the exact model snapshot that selected the protocol.
        # Do not predict again using a different/random row inside the explanation endpoint.
        snapshot = decision.get('model_snapshot') or model
        features = dict(decision.get('features') or {})
        required = list(MODEL_FEATURES)
        missing = [name for name in required if name not in features]
        if missing:
            return jsonify({'error': f'The captured selection is missing features: {missing}'}), 500
        expected_prediction = decision.get('selected_consensus')
        from consensus_explainer import explain_consensus
        result = explain_consensus(
            risk_assessment={}, transaction={},
            network_metrics={'input_sources': {name: decision.get('source', 'Existing dynamic selector input') for name in required}},
            active_nodes=len(getattr(network, 'nodes', {}) or {}),
            current_mode=active_mode,
            model=snapshot, scaler=scaler, label_encoder=label_encoder,
            training_frame=df_traffic, current_features=features,
        )
        # Preserve the actual selected label. If re-prediction does not match the captured
        # decision, flag it rather than silently presenting a different explanation.
        result['selected_consensus'] = expected_prediction
        result['current_consensus'] = active_mode
        result['selection_matches_current'] = expected_prediction == active_mode
        result['decision_method'] = 'Existing dynamic consensus selector (FL_SGD model)' 
        result['reason'] = f"The existing dynamic consensus selector selected {expected_prediction} for the captured input row. Module 4 explains this decision; it does not generate a new recommendation."
        result['algorithm_explanation'] = f"Actually selected by the existing selector: {expected_prediction}. Active blockchain mode now: {active_mode}."
        result['source_transaction'] = decision.get('transaction_id') or 'Most recent captured selection'
        result['selection_source'] = decision.get('source')
        result['training_label'] = decision.get('true_label')
        result['decision_timestamp'] = decision.get('timestamp')
        result['model_prediction_matches_captured'] = result.get('lime_explanation', {}).get('target_consensus') == expected_prediction
        result['method_note'] = 'LIME explains the exact model snapshot and feature row captured when the existing dynamic selector made its decision. Dataset values are not live network measurements unless separately instrumented. The explanation endpoint does not select or change consensus.'
        return jsonify(result)
    except Exception as exc:
        app.logger.exception('Module 4 actual-selection explanation failed')
        return jsonify({'error': 'Unable to explain the captured dynamic-consensus decision.', 'detail': str(exc)}), 500

@app.route('/api/set_consensus', methods=['POST'])
@persist_action
def set_consensus():
    mode = request.json.get('mode')
    success = network.set_consensus(mode)
    if success:
        return jsonify({'success': True, 'message': f'Consensus set to {mode}.'})
    else:
        return jsonify({'success': False, 'message': f'Invalid consensus mode: {mode}.'})

@app.route('/api/trigger', methods=['POST'])
@persist_action
def trigger_consensus():
    current_user = session.get('username', 'Anonymous')
    
    # Check if user is active (approved by community)
    load_network_state()
    if current_user not in network.active_users:
        return jsonify({'success': False, 'message': f'Only approved members can trigger consensus. Your account is still pending community approval.'}), 403
    
    result = network.run_consensus()
    return jsonify(result)

@app.route('/api/resolve', methods=['POST'])
@persist_action
def resolve_conflicts_route():
    result = network.resolve_conflicts()
    return jsonify(result)

# -----------------------------
# HTML TEMPLATES (Aesthetic UI/UX with Governance)
# -----------------------------
LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Land Registry | Secure Access</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top left, #f8fafc, #e2e8f0);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .login-container {
            background: #ffffff;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border: 1px solid #f1f5f9;
        }
        .input-field {
            transition: all 0.2s ease;
            border: 1px solid #e2e8f0;
        }
        .input-field:focus {
            border-color: #2563eb;
            ring: 2px;
            ring-color: #dbeafe;
            outline: none;
        }
        .btn-gov {
            background: #1e40af;
            transition: all 0.2s;
        }
        .btn-gov:hover {
            background: #1d4ed8;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2);
        }
    </style>
</head>
<body>
    <div class="login-container p-10 rounded-2xl w-[400px]">
        <div class="flex justify-center mb-6">
            <div class="bg-blue-600 p-3 rounded-xl shadow-lg shadow-blue-200">
                <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002-2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                </svg>
            </div>
        </div>
        
        <h1 class="text-2xl font-extrabold text-center text-slate-800 mb-1">
            Digital Land Registry
        </h1>
        <p class="text-center text-sm font-medium mb-8 text-slate-500">
            {% if is_login %} Official Land Records Portal {% else %} Register New Authority {% endif %}
        </p>
        
        <form method="POST" action="{% if is_login %}/login{% else %}/signup{% endif %}" class="space-y-5">
            <div>
                <label for="username" class="block text-xs font-bold uppercase tracking-wider mb-2 text-slate-500">Username</label>
                <input type="text" id="username" name="username" required minlength="3" placeholder="Enter ID"
                       class="w-full px-4 py-3 bg-slate-50 rounded-lg input-field text-slate-900 text-sm">
            </div>
            <div>
                <label for="password" class="block text-xs font-bold uppercase tracking-wider mb-2 text-slate-500">Password</label>
                <input type="password" id="password" name="password" required minlength="5" placeholder="••••••••"
                       class="w-full px-4 py-3 bg-slate-50 rounded-lg input-field text-slate-900 text-sm">
            </div>
            
            <button type="submit" class="btn-gov w-full text-white font-bold py-3 rounded-lg text-sm mt-2">
                {% if is_login %} Sign In to Registry {% else %} Request Access {% endif %}
            </button>
        </form>
        
        {% if message %}
        <div class="mt-4 p-3 bg-red-50 rounded-lg border border-red-100">
            <p class="text-center text-red-600 text-xs font-semibold">{{ message | safe }}</p>
        </div>
        {% endif %}

        <div class="mt-8 text-center">
            {% if is_login %}
            <a href="{{ url_for('signup') }}" class="text-xs font-bold text-blue-600 hover:text-blue-700 underline decoration-blue-200 underline-offset-4">
                New Official? Create Credentials
            </a>
            {% else %}
            <a href="{{ url_for('login') }}" class="text-xs font-bold text-blue-600 hover:text-blue-700 underline decoration-blue-200 underline-offset-4">
                Already Registered? Return to Sign In
            </a>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Digital Land Records Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            min-height: 100vh;
        }
        .header-gradient {
            background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%);
        }
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-radius: 12px;
        }
        .btn-action {
            background: #2563eb;
            color: white;
            font-weight: 600;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .btn-action:hover {
            background: #1d4ed8;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }
        .log-box {
            background: #f1f5f9;
            color: #334155;
            border: 1px solid #e2e8f0;
            overflow-y: scroll;
            height: 350px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.8rem;
            border-radius: 8px;
        }
        .hash-code {
            font-family: monospace;
            font-size: 0.75rem;
            color: #64748b;
            background: #f8fafc;
            padding: 2px 4px;
            border-radius: 4px;
        }
        .consensus-tag {
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.025em;
        }
        /* Refined Tags */
        .tag-pow { background: #fee2e2; color: #991b1b; }
        .tag-pos { background: #dcfce7; color: #166534; }
        .tag-raft { background: #fef9c3; color: #854d0e; }
        .tag-pbft { background: #dbeafe; color: #1e40af; }
        .tag-hotstuff { background: #ffedd5; color: #9a3412; }

        .block-verified {
            border-left: 4px solid #22c55e;
        }

        #ml-status {
            transition: all 0.3s ease;
        }
        
        .vote-btn-pending {
            background: #f59e0b;
            color: white;
        }
        /* ================= AI RISK EXPLANATION ================= */

        .risk-modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(4px);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .risk-modal {
            background: white;
            width: min(900px, 95vw);
            max-height: 90vh;
            overflow-y: auto;
            border-radius: 20px;
            box-shadow: 0 25px 60px rgba(0,0,0,0.25);
        }

        .risk-modal-header {
            padding: 22px 26px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .risk-modal-title {
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
        }

        .risk-close {
            width: 34px;
            height: 34px;
            border: none;
            border-radius: 50%;
            background: #f1f5f9;
            color: #475569;
            font-size: 20px;
            cursor: pointer;
        }

        .risk-close:hover {
            background: #e2e8f0;
        }

        .risk-modal-body {
            padding: 24px 26px;
        }

        .risk-summary {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 24px;
        }

        .risk-summary-card {
            padding: 16px;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }

        .risk-summary-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 6px;
        }

        .risk-summary-value {
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
        }

        .risk-section {
            margin-top: 24px;
        }

        .risk-section-title {
            font-size: 13px;
            font-weight: 800;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 12px;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .feature-item {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 14px;
        }

        .feature-name {
            font-size: 11px;
            color: #64748b;
            margin-bottom: 4px;
        }

        .feature-value {
            font-size: 14px;
            font-weight: 700;
            color: #0f172a;
        }

        .shap-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .shap-item {
            display: grid;
            grid-template-columns: 190px 90px 1fr;
            align-items: center;
            gap: 12px;
        }

        .shap-name {
            font-size: 11px;
            font-weight: 600;
            color: #334155;
        }

        .shap-value {
            font-family: monospace;
            font-size: 11px;
            font-weight: 700;
        }

        .shap-bar-container {
            height: 9px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
        }

        .shap-bar {
            height: 100%;
            border-radius: 10px;
        }

        .shap-positive {
            background: #ef4444;
        }

        .shap-negative {
            background: #22c55e;
        }

        .ai-interpretation {
            background: #f8fafc;
            border-left: 4px solid #6366f1;
            border-radius: 10px;
            padding: 15px 17px;
            font-size: 13px;
            line-height: 1.6;
            color: #475569;
        }

        .ai-warning {
            margin-top: 14px;
            padding: 12px 15px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 10px;
            font-size: 12px;
            color: #9a3412;
        }

        .view-risk-btn {
            margin-top: 12px;
            width: 100%;
            padding: 9px 12px;
            border: none;
            border-radius: 9px;
            background: #4f46e5;
            color: white;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            transition: 0.2s;
        }

        .view-risk-btn:hover {
            background: #4338ca;
            transform: translateY(-1px);
        }

        .risk-low {
            color: #16a34a;
        }

        .risk-medium {
            color: #d97706;
        }

        .risk-high {
            color: #dc2626;
        }

        @media (max-width: 640px) {
            .risk-summary {
                grid-template-columns: 1fr;
            }

            .feature-grid {
                grid-template-columns: 1fr;
            }

            .shap-item {
                grid-template-columns: 1fr;
                gap: 5px;
            }
        }
    </style>
</head>
<body class="p-4 sm:p-8">

    <div class="flex flex-col md:flex-row justify-between items-center mb-8 pb-6 border-b border-slate-200">
        <div class="flex items-center gap-3">
            <div class="bg-blue-600 p-2 rounded-lg">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path></svg>
            </div>
            <h1 class="text-3xl font-bold text-slate-900 tracking-tight">
                Digital Land Registry <span class="text-blue-600 font-medium text-lg block sm:inline sm:ml-2">Secure Ledger</span>
            </h1>
        </div>
        <div class="flex items-center gap-4 mt-4 md:mt-0 bg-white p-2 rounded-full shadow-sm border border-slate-100">
            <span class="text-sm px-3 text-slate-500 border-r border-slate-200">User: <span id="current-user" class="text-slate-900 font-semibold text-blue-600">...</span></span>
            <span id="ml-status" class="px-3 py-1 text-xs font-bold rounded-full">
                ML: <span id="ml-status-text">Checking...</span>
            </span>
            <a href="/logout" class="text-slate-400 hover:text-red-600 transition-colors mr-2">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path></svg>
            </a>
        </div>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div class="card p-5">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">ML Prediction</p>
            <div id="current-consensus" class="text-2xl font-bold text-slate-800">...</div>
            <div class="mt-2 text-xs text-slate-500 bg-slate-50 p-1 rounded">Difficulty Level: <span id="difficulty" class="font-bold text-blue-600">0</span></div>
        </div>
        <div class="card p-5">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Current Consensus</p>
            <div id="predicted-consensus" class="text-2xl font-bold text-indigo-600 italic">...</div>
            <div id="prediction-trigger" class="mt-2 text-xs text-slate-400 truncate">Source: System</div>
        </div>
        <div class="card p-5">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Network Nodes</p>
            <div id="active-user-count" class="text-2xl font-bold text-emerald-600">0</div>
            <div class="mt-2 text-xs text-slate-500">Pending Approvals: <span id="pending-user-count" class="text-orange-500 font-bold">0</span></div>
        </div>
        <div class="card p-5">
            <p class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Ledger Depth</p>
            <div id="chain-length" class="text-2xl font-bold text-blue-600">0</div>
            <div class="mt-2 text-xs text-slate-500">Unconfirmed TXs: <span id="pending-tx-count" class="text-red-500 font-bold">0</span></div>
        </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
        
        <div class="lg:col-span-8 space-y-8">
            
            <div class="card p-6">
                <h2 class="text-lg font-bold mb-4 text-slate-800 flex items-center gap-2">
                    <span class="w-2 h-6 bg-blue-600 rounded-full"></span>
                    Land Title Transfer
                </h2>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div class="space-y-1">
    <label class="text-xs font-semibold text-slate-500 ml-1">
        Land ID
    </label>

    <input
        type="text"
        id="land-id"
        placeholder="Property ID"
        class="w-full p-2.5 text-sm rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500">
</div>
                    <div class="space-y-1">
                        <label class="text-xs font-semibold text-slate-500 ml-1">Beneficiary Name</label>
                        <input type="text" id="new-owner" placeholder="Full Legal Name" class="w-full p-2.5 text-sm rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                    </div>
                    <div class="space-y-1">
                        <label class="text-xs font-semibold text-slate-500 ml-1">Valuation (INR)</label>
                        <input type="number" id="land-price" placeholder="Transaction Value" class="w-full p-2.5 text-sm rounded-lg border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                    </div>
                    <div class="space-y-1">
                        <label class="text-xs font-semibold text-slate-500 ml-1">Title Deed (PDF)</label>
                        <input type="file" id="land-pdf" accept="application/pdf" class="w-full text-xs text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer">
                    </div>
                </div>
                <button onclick="addLandTx()" class="btn-action w-full py-3 rounded-lg flex justify-center items-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    Authorize and Queue Transaction
                </button>
            </div>

            <div class="card p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-lg font-bold text-slate-800">Immutable Ledger</h2>
                    <span class="text-[10px] bg-slate-100 text-slate-500 px-2 py-1 rounded uppercase font-bold tracking-widest">Audited & Verified</span>
                </div>
                <div id="chain-view" class="log-box p-4 space-y-4">
                    </div>
            </div>
        </div>

        <div class="lg:col-span-4 space-y-8">
            
            <div class="card p-6 border-t-4 border-t-indigo-500">
                <h2 class="text-lg font-bold mb-4 text-slate-800">Validation Engine</h2>
                <div class="space-y-4">
                    <div>
                        <label class="text-xs font-semibold text-slate-500 mb-1 block">Override Protocol</label>
                        <select id="consensus" class="w-full p-2.5 text-sm rounded-lg bg-slate-50 border border-slate-200 outline-none focus:border-indigo-500">
                        </select>
                    </div>
                    <button onclick="triggerConsensus()" class="w-full py-3 rounded-lg bg-slate-900 text-white font-bold text-sm hover:bg-slate-800 transition-colors shadow-lg">
                        Forge Current Block
                    </button>
                    <button onclick="resolve()" class="w-full py-2 text-xs font-semibold text-slate-500 hover:text-indigo-600 transition-colors">
                        Synchronize Global State
                    </button>
                </div>
            </div>

            <!-- Module 4 moved below the two-column workspace for readable full-width layout. -->

            <div class="card p-6">
                <h2 class="text-lg font-bold mb-3 text-slate-800">Network Governance</h2>
                <div id="governance-view" class="min-h-[100px] border-y border-slate-50 py-3">
                    </div>
            </div>

            <div class="card p-6 bg-slate-50/50">
                <h2 class="text-md font-bold mb-3 text-slate-700">Participating Nodes</h2>
                <div id="node-list" class="space-y-2 mb-4">
                    </div>
                <div class="flex gap-2">
                    <input type="number" id="new-stake" placeholder="Stake" value="100" class="w-20 p-2 text-sm rounded border border-slate-200">
                    <button onclick="addNode()" class="flex-1 bg-white border border-slate-300 py-2 rounded text-xs font-bold hover:bg-slate-100 transition-colors">Register Node</button>
                </div>
            </div>
        </div>
    </div>

    <section id="module4-consensus" class="card p-5 md:p-7 mb-8 border-t-4 border-t-emerald-500 w-full">
        <div class="flex flex-col md:flex-row md:items-start md:justify-between gap-3 mb-6">
            <div>
                <p class="text-xs uppercase tracking-widest font-bold text-emerald-700">Module 4</p>
                <h2 class="text-2xl font-extrabold text-slate-900 mt-1">Explainable Dynamic Consensus</h2>
                <p class="text-sm text-slate-500 mt-2 max-w-3xl">Explain why the existing dynamic-consensus mechanism selected its actual protocol. Module 4 observes that decision and does not select a different protocol.</p>
            </div>
            <button onclick="loadConsensusExplanation()" class="shrink-0 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-bold px-4 py-2.5">Refresh explanation</button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
            <div class="rounded-xl border border-slate-200 bg-slate-50 p-4"><div class="text-xs uppercase tracking-wide font-bold text-slate-500">Dynamically selected protocol</div><div id="xai-consensus-selected" class="text-2xl font-extrabold text-emerald-700 mt-2">Analyzing…</div><div id="xai-consensus-reason" class="text-sm text-slate-700 mt-2 leading-relaxed">Loading explanation…</div></div>
            <div class="rounded-xl border border-slate-200 p-4"><div class="text-xs uppercase tracking-wide font-bold text-slate-500">Currently selected protocol</div><div id="xai-current-consensus" class="text-2xl font-extrabold text-indigo-700 mt-2">—</div><div id="xai-current-status" class="text-sm text-slate-600 mt-2">Checking whether the captured selection matches the active protocol.</div></div>
            <div class="rounded-xl border border-slate-200 p-4"><div class="text-xs uppercase tracking-wide font-bold text-slate-500">Transaction context</div><div class="grid grid-cols-2 gap-3 mt-3"><div><div class="text-xs text-slate-500">Fraud risk</div><div id="xai-risk" class="text-xl font-bold text-slate-900">—</div></div><div><div class="text-xs text-slate-500">Owner trust</div><div id="xai-trust" class="text-xl font-bold text-slate-900">—</div></div><div class="col-span-2"><div id="xai-network-summary" class="text-sm text-slate-600">Network metrics loading…</div><div id="xai-input-sources" class="text-xs text-slate-500 mt-2">Input provenance loading…</div></div></div></div>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-2 gap-5">
            <div class="rounded-xl border border-slate-200 p-4 md:p-5 min-w-0">
                <h3 class="text-lg font-bold text-slate-900">Why this protocol was recommended</h3>
                <p class="text-sm text-slate-500 mt-1 mb-4">These inputs are passed to the trained consensus-selection model; each input shows its source.</p>
                <div id="xai-consensus-factors" class="space-y-3"></div>
                <div id="xai-consensus-algorithm" class="text-sm text-slate-700 bg-indigo-50 border border-indigo-100 rounded-lg p-4 mt-4 leading-relaxed"></div>
                <div id="xai-consensus-note" class="text-xs text-slate-500 mt-3 leading-relaxed"></div>
            </div>
            <div class="rounded-xl border border-violet-200 bg-violet-50/40 p-4 md:p-5 min-w-0">
                <h3 class="text-lg font-bold text-violet-950">Why the selected consensus was chosen — LIME</h3>
                <p class="text-sm text-violet-800 mt-1">LIME locally approximates the trained consensus selector using representative project network-metrics rows.</p>
                <div id="xai-lime-target" class="text-sm font-semibold text-slate-800 mt-4">Waiting for explanation…</div>
                <div id="xai-lime-features" class="space-y-3 mt-3"></div>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4 text-xs text-slate-600"><div class="bg-white rounded-lg border border-violet-100 p-3"><strong>Positive weight</strong><p class="mt-1">Supports the selected class in the local surrogate.</p></div><div class="bg-white rounded-lg border border-violet-100 p-3"><strong>Negative weight</strong><p class="mt-1">Pushes against the selected class in the local surrogate.</p></div></div>
                <div id="xai-lime-warning" class="text-xs text-slate-500 mt-4 leading-relaxed"></div>
            </div>
        </div>

        <div class="mt-6">
            <div class="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2 mb-3"><div><h3 class="text-lg font-bold text-slate-900">Consensus protocol reference</h3><p class="text-sm text-slate-500 mt-1">Reference descriptions only. The highlighted protocol is the one selected by the existing dynamic-consensus pipeline.</p></div></div>
            <div id="xai-algorithm-options" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"><div class="text-sm text-slate-500">Loading protocol comparison…</div></div>
        </div>
    </section>

    <div class="card p-6 mb-8">
        <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-4">
            <div>
                <h2 class="text-lg font-bold text-slate-800">Project Transaction Graph + Blockchain Network</h2>
                <p class="text-xs text-slate-500 mt-1">The graph uses the mixed UK HM Land Registry dataset with synthetic owner/buyer/seller details. Search a User, Land, or Transaction to explore the connected records.</p>
            </div>
            <div class="flex flex-wrap gap-2">
                <input id="graph-postcode" class="p-2 text-xs rounded border border-slate-200" placeholder="Postcode filter (optional)">
                <input id="graph-property" class="p-2 text-xs rounded border border-slate-200" placeholder="Property ID filter (optional)">
                <button onclick="loadRealGraph({recentOnly:true})" class="bg-indigo-600 text-white px-4 py-2 rounded-lg text-xs font-bold">Show User Graph</button>
            </div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-2 mb-4 text-center text-xs">
            <div class="bg-slate-50 rounded p-2"><b id="graph-count-nodes">0</b><br>Nodes</div>
            <div class="bg-slate-50 rounded p-2"><b id="graph-count-edges">0</b><br>Edges</div>
            <div class="bg-slate-50 rounded p-2"><b id="graph-count-properties">0</b><br>Properties</div>
            <div class="bg-slate-50 rounded p-2"><b id="graph-count-transactions">0</b><br>Transactions</div>
            <div class="bg-slate-50 rounded p-2"><b id="graph-count-blocks">0</b><br>Blocks</div>
        </div>
        <div class="grid grid-cols-1 xl:grid-cols-4 gap-4 items-start">
            <div class="xl:col-span-3 space-y-4">
                <div class="border border-slate-200 rounded-xl overflow-hidden bg-slate-950">
                    <div id="real-graph-canvas" class="relative h-[620px] overflow-hidden">
                        <svg id="real-graph-svg" class="w-full h-full"></svg>
                        <div class="absolute top-3 left-3 z-10 flex flex-wrap gap-2">
                            <button onclick="graphZoom(1.25)" class="px-3 py-1.5 rounded-lg bg-white/90 text-slate-700 text-xs font-bold shadow">＋ Zoom</button>
                            <button onclick="graphZoom(0.8)" class="px-3 py-1.5 rounded-lg bg-white/90 text-slate-700 text-xs font-bold shadow">－ Zoom</button>
                            <button onclick="resetGraphView()" class="px-3 py-1.5 rounded-lg bg-white/90 text-slate-700 text-xs font-bold shadow">Fit / Reset View</button>
                        </div>
                        <div class="absolute bottom-3 left-3 z-10 px-3 py-2 rounded-lg bg-black/50 text-slate-300 text-[10px]">Scroll = zoom • Drag empty space = pan • Click node = inspect</div>
                        <div id="graph-empty" class="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">Load the graph, then search a node to explore its connected records.</div>
                    </div>
                </div>

                <!-- Selected-node details are deliberately below the graph.
                     This prevents the old short right-hand card from leaving a large
                     empty vertical area beside the 620px graph canvas. -->
                <div id="graph-selected-node" class="hidden bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h3 class="font-bold text-sm">Transaction / Node Details</h3>
                        <span class="text-[10px] text-slate-400">Selected graph node</span>
                    </div>
                    <div id="graph-selected-json" class="text-xs text-slate-600"></div>
                </div>
            </div>

            <div class="space-y-4">
                <div class="bg-indigo-50 rounded-xl p-4 border border-indigo-100">
                    <h3 class="font-bold text-sm mb-2 text-indigo-900">Find a Node</h3>
                    <p class="text-[10px] text-indigo-700 mb-2">Default view is a small overview of recent dataset records to keep the graph readable. Search a User, Land, or Transaction ID to explore the full matching history.</p>
                    <div class="flex gap-2 mb-2">
                        <select id="graph-search-type" class="w-32 p-2 text-xs rounded border border-indigo-200">
                            <option value="Owner" selected>User</option>
                            <option value="Property">Land</option>
                            <option value="Transaction">Transaction</option>
                        </select>
                        <input id="graph-search" class="flex-1 p-2 text-xs rounded border border-indigo-200" placeholder="e.g. Vishal" onkeydown="if(event.key==='Enter') focusGraphNode()">
                    </div>
                    <div class="flex gap-2 mb-2">
                        <select id="graph-depth" class="flex-1 p-2 text-xs rounded border border-indigo-200">
                            <option value="1" selected>Direct connections</option>
                            <option value="2">2-hop connections</option>
                            <option value="3">3-hop connections</option>
                        </select>
                        <button onclick="focusGraphNode()" class="flex-1 bg-indigo-600 text-white py-2 rounded text-xs font-bold">Find Node</button>
                    </div>
                    <button onclick="loadRealGraph({recentOnly:true})" class="w-full bg-white border border-indigo-200 py-2 rounded text-xs font-bold text-indigo-700">Show Recent Nodes Only</button>
                    <div id="graph-search-status" class="text-[10px] text-slate-500 mt-2"></div>
                </div>
                <div class="bg-slate-50 rounded-xl p-4">
                    <h3 class="font-bold text-sm mb-3">Add Graph Node</h3>
                    <input id="custom-node-type" class="w-full p-2 text-xs rounded border border-slate-200 mb-2" placeholder="Type e.g. Inspection">
                    <input id="custom-node-label" class="w-full p-2 text-xs rounded border border-slate-200 mb-2" placeholder="Label">
                    <button onclick="addGraphNode()" class="w-full bg-white border border-slate-300 py-2 rounded text-xs font-bold">Add Node</button>
                </div>
                <div class="bg-slate-50 rounded-xl p-4">
                    <h3 class="font-bold text-sm mb-2">Data rule</h3>
                    <p class="text-[11px] text-slate-500 leading-relaxed">The graph uses real HM Land Registry Price Paid records augmented with synthetic owner/buyer/seller details. Search queries the full mixed dataset; the visible default graph is capped for browser performance.</p>
                </div>
            </div>
        </div>
    </div>


    <div class="card p-6 mb-8">
        <h2 class="text-sm font-bold mb-4 text-slate-800 flex items-center gap-2">
    <span class="w-2 h-5 bg-blue-600 rounded-full"></span>
    Live Transaction Queue
</h2>

<div id="pending-tx-view" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
</div>

<!-- AI BLOCKED TRANSACTIONS -->
<div class="mt-8">

    <h2 class="text-sm font-bold mb-4 text-slate-800 flex items-center gap-2">
        <span class="w-2 h-5 bg-red-600 rounded-full"></span>
        AI Blocked Transactions
    </h2>

    <div id="blocked-tx-view"
         class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    </div>

</div>
    </div>

<script>
// Logic remains identical to your functional requirements, with minor visual tweaks in showMessage
function showMessage(title, message, isError = false) {
    const bgColor = isError ? 'bg-red-50' : 'bg-blue-50';
    const textColor = isError ? 'text-red-800' : 'text-blue-800';
    const borderColor = isError ? 'border-red-200' : 'border-blue-200';
    
    const msgBox = document.createElement('div');
    msgBox.className = `fixed top-6 right-6 z-50 p-4 rounded-xl shadow-2xl border ${bgColor} ${textColor} ${borderColor} transition-all duration-300 transform translate-y-0 max-w-sm`;
    msgBox.innerHTML = `
        <div class="flex items-start gap-3">
            <div class="flex-shrink-0">${isError ? '⚠️' : '✅'}</div>
            <div>
                <h3 class="font-bold text-sm">${title}</h3>
                <p class="text-xs opacity-90 mt-1 leading-relaxed">${message}</p>
            </div>
        </div>
    `;
    
    document.body.appendChild(msgBox);
    setTimeout(() => {
        msgBox.style.opacity = '0';
        msgBox.style.transform = 'translateY(-20px)'; // Moves UP instead of down when disappearing
        setTimeout(() => msgBox.remove(), 300);
    }, 4500);
}

// ... rest of your existing JS logic (refresh, render functions, etc.) goes here ...
// Ensure you update the rendering HTML inside JS to use the new class names if they changed.

let currentState = {};
let currentUsers = {};

function formatHash(hash) {
    if(!hash) return "N/A";
    return `${hash.substring(0, 8)}...${hash.substring(hash.length - 8)}`;
}

function getConsensusTag(mode) {
    const modeClass = mode.toLowerCase().replace(/[^a-z]/g, '');
    return `<span class="consensus-tag tag-${modeClass}">${mode}</span>`;
}

function renderMLStatus(mlReady) {
    const mlStatusEl = document.getElementById('ml-status-text');
    const container = document.getElementById('ml-status');
    if (mlReady) {
        mlStatusEl.textContent = 'SYSTEM OPTIMIZED';
        container.className = 'px-3 py-1 text-xs font-bold rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200';
    } else {
        mlStatusEl.textContent = 'STATIC MODE';
        container.className = 'px-3 py-1 text-xs font-bold rounded-full bg-slate-100 text-slate-500 border border-slate-200';
    }
}

function renderNodeList(nodes) {
    const listEl = document.getElementById('node-list');
    let html = '';
    for (const name in nodes) {
        const node = nodes[name];
        html += `
            <div class="flex justify-between items-center w-full py-2 px-3 bg-white rounded-lg border border-slate-100 text-sm shadow-sm">
                <span class="font-semibold text-slate-700">${name}</span>
                <div class="flex items-center gap-3">
                    <span class="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">Stake: ${node.stake}</span>
                    <button onclick="removeNode('${name}')" class="text-slate-300 hover:text-red-500 transition-colors">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 000 2h6a1 1 0 100-2H7z" clip-rule="evenodd"></path></svg>
                    </button>
                </div>
            </div>
        `;
    }
    listEl.innerHTML = html || '<p class="text-xs text-slate-400 text-center py-2">No active nodes</p>';
}
function renderChain(chain, difficulty) {
    const chainViewEl = document.getElementById('chain-view');
    chainViewEl.innerHTML = '';
    document.getElementById('chain-length').textContent = chain.length;

    chain.slice().reverse().forEach(block => {
        const blockHash = block.hash || formatHash(block.previous_hash + block.index);
        const isVerified = blockHash.startsWith('0'.repeat(difficulty));
        const verifiedClass = isVerified ? 'block-verified' : 'border-l-4 border-slate-200';

        const tx = block.transactions?.[0] || {};
        const consensus = tx.predicted_consensus || block.consensus || 'N/A';

        const landDetails = tx.old_owner ? `
            <div class="mt-3 grid grid-cols-2 gap-2 text-[11px] bg-slate-50 p-3 rounded border border-slate-100">
                <div>
                    <span class="text-slate-400 block uppercase font-bold text-[9px]">Previous Owner</span>
                    ${tx.old_owner}
                </div>
                <div>
                    <span class="text-slate-400 block uppercase font-bold text-[9px]">New Owner</span>
                    ${tx.new_owner}
                </div>
                <div class="col-span-2">
                    <span class="text-slate-400 block uppercase font-bold text-[9px]">Land Value</span>
                    ₹${Number(tx.land_price || 0).toLocaleString("en-IN")}
                </div>
                <div>
                    <span class="text-slate-400 block uppercase font-bold text-[9px]">Land ID</span>
                    ${tx.land_id}
                </div>
                <div class="col-span-2">
                    <span class="text-slate-400 block uppercase font-bold text-[9px]">Document Hash</span>
                    <span class="font-mono break-all text-slate-600">${formatHash(tx.document_hash)}</span>
                </div>
                <div class="col-span-2">
                    <span class="text-slate-400 block uppercase font-bold text-[9px]">Transaction ID</span>
                    <span class="font-mono break-all text-slate-600">${formatHash(tx.id)}</span>
                </div>
            </div>
        ` : '';

        const cryptoDetails = tx.pqc_signature ? `
            <details class="mt-3 text-[11px] bg-slate-50 rounded border border-slate-100">
                <summary class="cursor-pointer px-3 py-2 font-bold text-slate-600">
                    Cryptographic Proofs (PQC)
                </summary>
                <div class="p-3 space-y-2 font-mono text-[10px] text-slate-600 break-all">
                    <div>
                        <span class="text-slate-400 font-bold">SIGNATURE</span><br>
                        ${tx.pqc_signature.slice(0, 120)}…
                    </div>
                    <div>
                        <span class="text-slate-400 font-bold">PUBLIC KEY</span><br>
                        ${tx.pqc_public_key.slice(0, 120)}…
                    </div>
                </div>
            </details>
        ` : '';

        chainViewEl.innerHTML += `
            <div class="p-4 mb-4 rounded-xl bg-white border border-slate-200 shadow-sm ${verifiedClass}">
                <div class="flex justify-between items-center mb-3">
                    <span class="font-bold text-slate-800">Block #${block.index}</span>
                    <span class="text-[10px] text-slate-400 font-mono">
                        ${new Date((tx.timestamp || block.timestamp) * 1000).toLocaleString()}
                    </span>
                </div>

                <div class="flex gap-2 mb-3">
                    ${getConsensusTag(consensus)}
                    <span class="text-[10px] py-1 px-2 bg-slate-100 rounded text-slate-600 font-bold uppercase">
                        Proposer: ${block.proposer || 'System'}
                    </span>
                </div>

                <div class="space-y-1 text-[10px] font-mono">
                    <div class="flex justify-between text-slate-400">
                        <span>BLOCK HASH</span>
                        <span class="text-slate-600">${blockHash}</span>
                    </div>
                    <div class="flex justify-between text-slate-400">
                        <span>PREVIOUS</span>
                        <span class="text-slate-600">${formatHash(block.previous_hash)}</span>
                    </div>
                </div>

                ${landDetails}
                ${cryptoDetails}
            </div>
        `;
    });
}


function renderPendingTx(txList) {
    const pendingTxViewEl = document.getElementById('pending-tx-view');
    pendingTxViewEl.innerHTML = '';

    if (txList.length === 0) {
        pendingTxViewEl.innerHTML =
            '<div class="col-span-full py-8 text-center text-slate-400 text-sm italic">No transactions currently in queue...</div>';
        return;
    }

    txList.forEach((tx, index) => {

        const risk = tx.risk_assessment || {};

        const riskLevel = risk.risk_level || "UNKNOWN";
        const riskScore = risk.risk_score ?? 0;
        const recommendation = risk.recommendation || "N/A";

        let riskClass = "risk-medium";

        if (riskLevel === "LOW") {
            riskClass = "risk-low";
        } else if (riskLevel === "HIGH") {
            riskClass = "risk-high";
        }

        pendingTxViewEl.innerHTML += `
            <div class="p-4 rounded-lg bg-white border border-slate-200 shadow-sm">

                <div class="flex justify-between items-start mb-3">

                    <div>
                        <span class="text-blue-600 font-mono text-[10px]">
                            TX_${formatHash(tx.id)}
                        </span>

                        <p class="text-sm font-bold text-slate-800 mt-1">
                            Land ID: ${tx.land_id}
                        </p>
                    </div>

                    <span class="text-slate-800 font-bold text-sm">
                        ₹${Number(tx.land_price).toLocaleString("en-IN")}
                    </span>

                </div>

                <div class="text-[11px] space-y-1 mb-3">
                    <p>
                        <span class="text-slate-500">FROM:</span>
                        ${tx.old_owner}
                    </p>

                    <p>
                        <span class="text-slate-500">TO:</span>
                        ${tx.new_owner}
                    </p>
                </div>

                <div class="flex justify-between items-center 
                            bg-slate-50 rounded-lg px-3 py-2">

                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold">
                            AI Risk
                        </p>

                        <p class="text-sm font-extrabold ${riskClass}">
                            ${riskLevel}
                        </p>
                    </div>

                    <div class="text-right">
                        <p class="text-[9px] text-slate-500 uppercase font-bold">
                            Score
                        </p>

                        <p class="text-sm font-extrabold ${riskClass}">
                            ${Number(riskScore).toFixed(2)} / 100
                        </p>
                    </div>

                </div>

                <button
                    class="view-risk-btn"
                    onclick="showRiskExplanation(${index})">

                    🔍 View AI Explanation

                </button>

            </div>
        `;
    });

    window.currentPendingTransactions = txList;
}
function renderBlockedTx(txList) {

    const container = document.getElementById('blocked-tx-view');

    if (!container) return;

    container.innerHTML = '';

    if (!txList || txList.length === 0) {
        container.innerHTML =
            '<div class="col-span-full py-8 text-center text-slate-400 text-sm italic">No AI-blocked transactions...</div>';

        return;
    }

    window.currentBlockedTransactions = txList;

    txList.forEach((item, index) => {

        const tx = item.transaction || {};
        const risk = item.risk_assessment || {};

        container.innerHTML += `
            <div class="p-4 rounded-lg bg-white border border-red-200 shadow-sm">

                <div class="flex justify-between items-start mb-3">

                    <div>
                        <span class="text-red-600 font-mono text-[10px]">
                            AI BLOCKED
                        </span>

                        <p class="text-sm font-bold text-slate-800 mt-1">
                            Land ID: ${tx.land_id || "N/A"}
                        </p>
                    </div>

                    <span class="text-slate-800 font-bold text-sm">
                        ₹${Number(tx.land_price || 0).toLocaleString("en-IN")}
                    </span>

                </div>

                <div class="text-[11px] space-y-1 mb-3">

                    <p>
                        <span class="text-slate-500">FROM:</span>
                        ${tx.old_owner || "N/A"}
                    </p>

                    <p>
                        <span class="text-slate-500">TO:</span>
                        ${tx.new_owner || "N/A"}
                    </p>

                </div>

                <div class="flex justify-between items-center bg-red-50 rounded-lg px-3 py-2">

                    <div>
                        <p class="text-[9px] text-slate-500 uppercase font-bold">
                            AI Risk
                        </p>

                        <p class="text-sm font-extrabold risk-high">
                            HIGH
                        </p>
                    </div>

                    <div class="text-right">

                        <p class="text-[9px] text-slate-500 uppercase font-bold">
                            Score
                        </p>

                        <p class="text-sm font-extrabold risk-high">
                            ${Number(risk.risk_score || 0).toFixed(2)} / 100
                        </p>

                    </div>

                </div>

                <div class="mt-3 p-3 bg-red-50 rounded-lg">

                    <p class="text-[11px] text-red-700">
                        ❌ This transaction was blocked because the AI detected HIGH risk.
                    </p>

                </div>

                <button
                    class="view-risk-btn"
                    onclick="showRiskExplanation(${index},'blocked')">

                    🔍 View AI Explanation

                </button>

            </div>
        `;
    });
}



function showRiskExplanation(index,type="pending") {

    let tx;
    let risk;

    if (type === "blocked") {

        const blockedItem =
            window.currentBlockedTransactions[index];

        if (!blockedItem) {
            alert("Blocked transaction is not available.");
            return;
        }

        tx = blockedItem.transaction || {};
        risk = blockedItem.risk_assessment || {};

    } else {

        tx = window.currentPendingTransactions[index];

        if (!tx || !tx.risk_assessment) {
            alert("AI risk assessment is not available for this transaction.");
            return;
        }

        risk = tx.risk_assessment;
    }

    const features = risk.features || {};
    const shap = risk.shap_explanation || {};


    const riskLevel = risk.risk_level || "UNKNOWN";
    const recommendation = risk.recommendation || "N/A";
    const riskScore = Number(risk.risk_score || 0);
    const fraudProbability = Number(risk.fraud_probability || 0);

    let riskClass = "risk-medium";

    if (riskLevel === "LOW") {
        riskClass = "risk-low";
    } else if (riskLevel === "HIGH") {
        riskClass = "risk-high";
    }

    const featureLabels = {
        land_price: "Land Price",
        land_transaction_count: "Previous Land Transactions",
        document_duplicate_count: "Document Duplicates",
        seller_transaction_count: "Seller Transactions",
        buyer_transaction_count: "Buyer Transactions",
        price_deviation: "Price Deviation",
        rapid_transfer_feature: "Rapid Transfer"
    };

    function formatFeatureValue(name, value) {

        if (name === "land_price") {
            return "₹" + Number(value).toLocaleString("en-IN");
        }

        if (name === "price_deviation") {
            return Number(value).toFixed(2) + "×";
        }

        if (name === "rapid_transfer_feature") {
            return Number(value) === 1
                ? "Detected"
                : "Not Detected";
        }

        return Number(value).toLocaleString("en-IN");
    }

    let featureHTML = "";

    Object.keys(featureLabels).forEach(name => {

        if (features[name] === undefined) return;

        featureHTML += `
            <div class="feature-item">

                <div class="feature-name">
                    ${featureLabels[name]}
                </div>

                <div class="feature-value">
                    ${formatFeatureValue(name, features[name])}
                </div>

            </div>
        `;
    });


    // ---------------- SHAP ----------------

    const shapEntries = Object.entries(shap);

    let maxAbsShap = 0;

    shapEntries.forEach(([name, value]) => {
        maxAbsShap = Math.max(
            maxAbsShap,
            Math.abs(Number(value))
        );
    });

    if (maxAbsShap === 0) {
        maxAbsShap = 1;
    }

    let shapHTML = "";

    shapEntries.forEach(([name, value]) => {

        const shapValue = Number(value);

        const percentage =
            Math.min(
                100,
                (Math.abs(shapValue) / maxAbsShap) * 100
            );

        const impactClass =
            shapValue >= 0
                ? "shap-positive"
                : "shap-negative";

        const impactText =
            shapValue >= 0
                ? "↑ Risk"
                : "↓ Risk";

        shapHTML += `
            <div class="shap-item">

                <div class="shap-name">
                    ${featureLabels[name] || name}
                </div>

                <div class="shap-value ${shapValue >= 0 ? 'risk-high' : 'risk-low'}">
                    ${shapValue >= 0 ? '+' : ''}
                    ${shapValue.toFixed(6)}
                    <span style="font-family:inherit; font-size:9px;">
                        ${impactText}
                    </span>
                </div>

                <div class="shap-bar-container">

                    <div
                        class="shap-bar ${impactClass}"
                        style="width:${percentage}%">
                    </div>

                </div>

            </div>
        `;
    });


    // ---------------- INTERPRETATION ----------------

    const positiveFactors = shapEntries
        .filter(([name, value]) => Number(value) > 0)
        .sort((a, b) => Number(b[1]) - Number(a[1]));

    let interpretation =
        "The AI model evaluated the transaction using the available land transaction features.";

    if (positiveFactors.length > 0) {

        const strongest = positiveFactors[0][0];

        const explanations = {

            land_price:
                "The land price contributed to the model's risk assessment.",

            land_transaction_count:
                "The number of previous transactions involving this land contributed to the risk assessment.",

            document_duplicate_count:
                "Repeated document information contributed to the risk assessment.",

            seller_transaction_count:
                "The seller's transaction history contributed to the risk assessment.",

            buyer_transaction_count:
                "The buyer's transaction history contributed to the risk assessment.",

            price_deviation:
                "The difference between the current price and the previous land transaction price was a major contributor to the risk assessment.",

            rapid_transfer_feature:
                "The short time interval between land transfers contributed to the risk assessment."
        };

        interpretation =
            explanations[strongest] ||
            interpretation;
    }

    // ---------------- MODAL ----------------

    const modal = document.createElement("div");

    modal.className = "risk-modal-overlay";

    modal.innerHTML = `

        <div class="risk-modal">

            <!-- HEADER -->

            <div class="risk-modal-header">

                <div>

                    <div class="risk-modal-title">
                        🤖 AI Risk Explanation
                    </div>

                    <div class="text-[11px] text-slate-500 mt-1">
                        Land ID: ${tx.land_id}
                        &nbsp; • &nbsp;
                        TX_${formatHash(tx.id)}
                    </div>

                </div>

                <button
                    class="risk-close"
                    onclick="this.closest('.risk-modal-overlay').remove()">

                    ×

                </button>

            </div>


            <!-- BODY -->

            <div class="risk-modal-body">


                <!-- SUMMARY -->

                <div class="risk-summary">

                    <div class="risk-summary-card">

                        <div class="risk-summary-label">
                            Risk Level
                        </div>

                        <div class="risk-summary-value ${riskClass}">
                            ${riskLevel}
                        </div>

                    </div>


                    <div class="risk-summary-card">

                        <div class="risk-summary-label">
                            Risk Score
                        </div>

                        <div class="risk-summary-value ${riskClass}">
                            ${riskScore.toFixed(2)} / 100
                        </div>

                    </div>


                    <div class="risk-summary-card">

                        <div class="risk-summary-label">
                            Fraud Probability
                        </div>

                        <div class="risk-summary-value ${riskClass}">
                            ${(fraudProbability * 100).toFixed(2)}%
                        </div>

                    </div>

                </div>


                <!-- RECOMMENDATION -->

                <div class="risk-summary-card mb-5">

                    <div class="risk-summary-label">
                        AI Recommendation
                    </div>

                    <div class="risk-summary-value ${riskClass}">
                        ${recommendation}
                    </div>

                </div>


                <!-- FEATURES -->

                <div class="risk-section">

                    <div class="risk-section-title">
                         Features Used by GNN
                    </div>

                    <div class="feature-grid">

                        ${featureHTML}

                    </div>

                </div>


                <!-- SHAP -->

                <div class="risk-section">

                    <div class="risk-section-title">
                        🔍 SHAP Explanation
                    </div>

                    <div class="shap-list">

                        ${shapHTML}

                    </div>

                    <div class="text-[10px] text-slate-400 mt-3">
                        Positive SHAP values increase the model's
                        fraud-risk prediction. Negative SHAP values
                        decrease it.
                    </div>

                </div>


                <!-- INTERPRETATION -->

                <div class="risk-section">

                    <div class="risk-section-title">
                        💡 AI Interpretation
                    </div>

                    <div class="ai-interpretation">

                        ${interpretation}

                    </div>

                    <div class="ai-warning">

                        ⚠️ <strong>Human Review Required:</strong>
                        The AI recommendation is advisory.
                        The final transaction decision should be
                        made by the authorized user.

                    </div>

                </div>


            </div>

        </div>
    `;

    document.body.appendChild(modal);


    // Close when clicking outside modal

    modal.addEventListener("click", function(event) {

        if (event.target === modal) {
            modal.remove();
        }

    });

}

function renderGovernance(users) {
    const governanceEl = document.getElementById('governance-view');
    document.getElementById('active-user-count').textContent = users.active_count;
    document.getElementById('pending-user-count').textContent = users.pending_users.length;
    
    if (users.pending_users.length === 0) {
        governanceEl.innerHTML = '<p class="text-xs text-slate-400 text-center py-4 italic">No pending authorizations</p>';
        return;
    }

    let html = `<p class="text-[10px] font-bold text-orange-600 mb-3 uppercase tracking-wider">Pending Peer-Review (${users.majority_threshold} votes required)</p>`;
    users.pending_users.forEach(pUser => {
        const isVoted = pUser.voted_by_current;
        html += `
            <div class="flex justify-between items-center py-3 border-b border-slate-50 last:border-0">
                <div>
                    <p class="text-sm font-bold text-slate-700">${pUser.username}</p>
                    <p class="text-[10px] text-slate-400">Progress: ${pUser.vote_count}/${users.active_count}</p>
                </div>
                <button onclick="voteUser('${pUser.username}')" 
                        class="text-[10px] font-bold px-4 py-1.5 rounded-full transition-all ${isVoted ? 'bg-slate-100 text-slate-400 cursor-not-allowed' : 'bg-orange-500 text-white hover:bg-orange-600'}" 
                        ${isVoted ? 'disabled' : ''}>
                    ${isVoted ? 'VERIFIED' : 'APPROVE'}
                </button>
            </div>
        `;
    });
    governanceEl.innerHTML = html;
}

// ... the rest of the existing API trigger functions remain unchanged ...
let latestRealGraph = null;
let graphNodes = [];
let graphEdges = [];
let graphAnimation = null;
let graphScale = 1;
let graphOffsetX = 0;
let graphOffsetY = 0;
let graphPan = null;
let graphDraggingNode = null;
let graphWorld = null;

function graphColor(type) {
    const colors = {
        Property: '#38bdf8',
        Transaction: '#a78bfa',
        Owner: '#fb7185',
        Area: '#fbbf24',
        Postcode: '#34d399',
        BlockchainLand: '#22c55e',
        BlockchainTransaction: '#fb7185',
        Block: '#f59e0b',
        NetworkNode: '#14b8a6',
        Custom: '#e2e8f0'
    };
    return colors[type] || '#94a3b8';
}

function graphNodeRadius(n) {
    if (n.id === window.graphFocusNodeId) return 24;
    if (n.type === 'Property' || n.type === 'BlockchainLand') return 17;
    if (n.type === 'Owner') return 14;
    return 11;
}

function graphLabel(n) {
    const text = String(n.label || n.id || '');
    return text.length > 34 ? text.slice(0, 34) + '…' : text;
}

function stopGraphAnimation() {
    if (graphAnimation) {
        cancelAnimationFrame(graphAnimation);
        graphAnimation = null;
    }
}

function updateGraphWorldTransform() {
    if (graphWorld) {
        graphWorld.setAttribute('transform',
            `translate(${graphOffsetX},${graphOffsetY}) scale(${graphScale})`);
    }
}

function graphZoom(factor) {
    const svg = document.getElementById('real-graph-svg');
    if (!svg) return;
    const oldScale = graphScale;
    graphScale = Math.max(0.25, Math.min(5, graphScale * factor));

    // Zoom around the centre of the canvas instead of jumping to the top-left.
    const rect = svg.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    graphOffsetX = cx - (cx - graphOffsetX) * (graphScale / oldScale);
    graphOffsetY = cy - (cy - graphOffsetY) * (graphScale / oldScale);
    updateGraphWorldTransform();
}

function fitGraphToView(nodes = graphNodes) {
    const svg = document.getElementById('real-graph-svg');
    if (!svg || !graphWorld || !nodes || !nodes.length) return;

    const rect = svg.getBoundingClientRect();
    const width = rect.width || 900;
    const height = rect.height || 620;
    const xs = nodes.map(n => Number(n.x)).filter(Number.isFinite);
    const ys = nodes.map(n => Number(n.y)).filter(Number.isFinite);
    if (!xs.length || !ys.length) return;

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = Math.max(120, maxX - minX);
    const spanY = Math.max(120, maxY - minY);
    const padding = 70;

    // Fit the finished force layout into the visible canvas without changing
    // the actual node positions used by the simulation.
    const fitScale = Math.max(0.25, Math.min(1.35,
        Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY)));

    graphScale = fitScale;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    graphOffsetX = width / 2 - centerX * graphScale;
    graphOffsetY = height / 2 - centerY * graphScale;
    updateGraphWorldTransform();
}

function resetGraphView() {
    fitGraphToView(graphNodes);
}

function createGraphSvgElement(tag, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
}

function renderGraphSubset(nodes, edges, titleText = '') {
    const svg = document.getElementById('real-graph-svg');
    const empty = document.getElementById('graph-empty');
    if (!svg) return;

    stopGraphAnimation();
    svg.innerHTML = '';
    graphWorld = null;

    if (!nodes || !nodes.length) {
        if (empty) empty.style.display = 'flex';
        return;
    }
    if (empty) empty.style.display = 'none';

    const rect = svg.getBoundingClientRect();
    const width = Math.max(700, rect.width || 900);
    const height = Math.max(500, rect.height || 620);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

    graphScale = 1;
    graphOffsetX = 0;
    graphOffsetY = 0;

    // Clone data so the original API response is never mutated.
    graphNodes = nodes.map((n, i) => ({
        ...n,
        x: n.x ?? width / 2 + Math.cos(i * 2.399) * (80 + (i % 7) * 22),
        y: n.y ?? height / 2 + Math.sin(i * 2.399) * (70 + (i % 7) * 20),
        vx: 0,
        vy: 0
    }));
    const idSet = new Set(graphNodes.map(n => n.id));
    graphEdges = edges.filter(e => idSet.has(e.source) && idSet.has(e.target)).map(e => ({...e}));
    const byId = new Map(graphNodes.map(n => [n.id, n]));
    const focusId = window.graphFocusNodeId;

    graphWorld = createGraphSvgElement('g', {class:'graph-world'});
    svg.appendChild(graphWorld);

    const edgeLayer = createGraphSvgElement('g', {class:'graph-edges'});
    const nodeLayer = createGraphSvgElement('g', {class:'graph-nodes'});
    graphWorld.appendChild(edgeLayer);
    graphWorld.appendChild(nodeLayer);

    // Draw edges first so nodes always sit above them.
    graphEdges.forEach(e => {
        const line = createGraphSvgElement('line', {
            stroke: (e.source === focusId || e.target === focusId) ? '#cbd5e1' : '#64748b',
            'stroke-width': (e.source === focusId || e.target === focusId) ? '2.8' : '1.1',
            'stroke-opacity': (e.source === focusId || e.target === focusId) ? '0.9' : '0.38'
        });
        edgeLayer.appendChild(line);
        e.__el = line;
    });

    graphNodes.forEach(n => {
        const g = createGraphSvgElement('g', {'data-node-id': n.id});
        g.style.cursor = 'grab';

        const circle = createGraphSvgElement('circle', {
            r: graphNodeRadius(n),
            fill: graphColor(n.type),
            stroke: n.id === focusId ? '#ffffff' : '#020617',
            'stroke-width': n.id === focusId ? '4' : '2'
        });
        g.appendChild(circle);

        // Only show labels for the focused node. Other labels appear on hover.
        const text = createGraphSvgElement('text', {
            x: '0',
            y: String(graphNodeRadius(n) + 17),
            'text-anchor': 'middle',
            fill: '#f8fafc',
            'font-size': n.id === focusId ? '14' : '11',
            'font-weight': n.id === focusId ? '800' : '600',
            'paint-order': 'stroke',
            stroke: '#020617',
            'stroke-width': '3',
            visibility: n.id === focusId ? 'visible' : 'hidden'
        });
        text.textContent = graphLabel(n);
        g.appendChild(text);

        const badge = createGraphSvgElement('text', {
            x: '0', y: '4', 'text-anchor': 'middle',
            fill: '#020617', 'font-size': '8', 'font-weight': '800',
            'pointer-events': 'none'
        });
        const initials = String(n.type || '?').slice(0, 1).toUpperCase();
        badge.textContent = initials;
        g.appendChild(badge);

        g.addEventListener('mouseenter', () => {
            circle.setAttribute('stroke', '#ffffff');
            circle.setAttribute('stroke-width', '3.5');
            text.setAttribute('visibility', 'visible');
            document.getElementById('graph-search-status').textContent =
                `${n.type}: ${n.label || n.id}`;
        });
        g.addEventListener('mouseleave', () => {
            circle.setAttribute('stroke', n.id === window.graphFocusNodeId ? '#ffffff' : '#020617');
            circle.setAttribute('stroke-width', n.id === window.graphFocusNodeId ? '4' : '2');
            if (n.id !== window.graphFocusNodeId) text.setAttribute('visibility', 'hidden');
        });
        g.addEventListener('click', (event) => {
            event.stopPropagation();
            selectAndFocusGraphNode(n.id);
        });

        // Native SVG pointer dragging — no D3 dependency required.
        g.addEventListener('pointerdown', (event) => {
            event.stopPropagation();
            graphDraggingNode = {
                node: n,
                startX: event.clientX,
                startY: event.clientY,
                nodeX: n.x,
                nodeY: n.y
            };
            g.setPointerCapture(event.pointerId);
            g.style.cursor = 'grabbing';
        });
        g.addEventListener('pointermove', (event) => {
            if (!graphDraggingNode || graphDraggingNode.node !== n) return;
            const dx = (event.clientX - graphDraggingNode.startX) / graphScale;
            const dy = (event.clientY - graphDraggingNode.startY) / graphScale;
            n.x = graphDraggingNode.nodeX + dx;
            n.y = graphDraggingNode.nodeY + dy;
            n.vx = 0; n.vy = 0;
        });
        g.addEventListener('pointerup', () => {
            graphDraggingNode = null;
            g.style.cursor = 'grab';
        });

        nodeLayer.appendChild(g);
        n.__el = g;
        n.__circle = circle;
        n.__text = text;
    });

    svg.addEventListener('click', clearGraphSelection, {once:true});
    installGraphCanvasControls();

    // Lightweight force simulation. This gives the "floating nodes" look without
    // requiring an external JavaScript library.
    const centerX = width / 2;
    const centerY = height / 2;
    const iterations = 180;
    let tick = 0;

    function step() {
        if (!graphNodes.length) return;
        tick++;

        // Repulsion.
        for (let i = 0; i < graphNodes.length; i++) {
            const a = graphNodes[i];
            for (let j = i + 1; j < graphNodes.length; j++) {
                const b = graphNodes[j];
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let dist2 = dx * dx + dy * dy;
                if (dist2 < 25) dist2 = 25;
                const dist = Math.sqrt(dist2);
                const force = Math.min(1800 / dist2, 0.8);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                a.vx -= fx; a.vy -= fy;
                b.vx += fx; b.vy += fy;
            }
        }

        // Spring forces along real graph edges.
        graphEdges.forEach(e => {
            const a = byId.get(e.source);
            const b = byId.get(e.target);
            if (!a || !b) return;
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            const dist = Math.max(1, Math.sqrt(dx*dx + dy*dy));
            const desired = focusId ? 125 : 105;
            const force = (dist - desired) * 0.0035;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            a.vx += fx; a.vy += fy;
            b.vx -= fx; b.vy -= fy;
        });

        // Gentle pull to centre + damping.
        graphNodes.forEach(n => {
            n.vx += (centerX - n.x) * 0.0009;
            n.vy += (centerY - n.y) * 0.0009;
            n.vx *= 0.82;
            n.vy *= 0.82;
            if (!graphDraggingNode || graphDraggingNode.node !== n) {
                n.x += n.vx;
                n.y += n.vy;
            }
            const pad = 35;
            n.x = Math.max(pad, Math.min(width-pad, n.x));
            n.y = Math.max(pad, Math.min(height-pad, n.y));
        });

        graphEdges.forEach(e => {
            const a = byId.get(e.source), b = byId.get(e.target);
            if (a && b && e.__el) {
                e.__el.setAttribute('x1', a.x);
                e.__el.setAttribute('y1', a.y);
                e.__el.setAttribute('x2', b.x);
                e.__el.setAttribute('y2', b.y);
            }
        });
        graphNodes.forEach(n => {
            if (n.__el) n.__el.setAttribute('transform', `translate(${n.x},${n.y})`);
        });

        if (tick < iterations || graphDraggingNode) {
            graphAnimation = requestAnimationFrame(step);
        } else {
            graphAnimation = null;
            // Automatically centre and scale the completed layout so the graph
            // does not sit in a small corner of the canvas.
            fitGraphToView(graphNodes);
        }
    }
    step();

    document.getElementById('graph-search-status').textContent =
        titleText || `${nodes.length} floating nodes shown. Drag nodes, pan the canvas or scroll to zoom.`;
}

function installGraphCanvasControls() {
    const svg = document.getElementById('real-graph-svg');
    if (!svg || svg.dataset.controlsInstalled === '1') return;
    svg.dataset.controlsInstalled = '1';

    svg.style.cursor = 'grab';
    svg.addEventListener('wheel', (event) => {
        event.preventDefault();
        graphZoom(event.deltaY < 0 ? 1.12 : 0.89);
    }, {passive:false});

    svg.addEventListener('pointerdown', (event) => {
        if (event.target.closest('[data-node-id]')) return;
        graphPan = {x:event.clientX, y:event.clientY, ox:graphOffsetX, oy:graphOffsetY};
        svg.style.cursor = 'grabbing';
        svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener('pointermove', (event) => {
        if (!graphPan) return;
        graphOffsetX = graphPan.ox + (event.clientX - graphPan.x);
        graphOffsetY = graphPan.oy + (event.clientY - graphPan.y);
        updateGraphWorldTransform();
    });
    svg.addEventListener('pointerup', () => {
        graphPan = null;
        svg.style.cursor = 'grab';
    });
    svg.addEventListener('pointercancel', () => {
        graphPan = null;
        svg.style.cursor = 'grab';
    });
}

function clearGraphSelection() {
    const status = document.getElementById('graph-search-status');
    if (status) status.textContent = 'Search a node to explore its connected neighbourhood.';
}

function selectAndFocusGraphNode(nodeId) {
    if (!latestRealGraph) return;
    const depth = parseInt(document.getElementById('graph-depth').value || '2', 10);
    renderConnectedNode(nodeId, depth);
}

function renderRealGraph(graph) {
    latestRealGraph = graph;

    // The backend returns counts by exact node type (Owner, Property,
    // Transaction, BlockchainTransaction, Block, etc.).  The old UI only
    // looked for the exact strings Property/Transaction, so the cards could
    // incorrectly show 0 even when those nodes were present. Count from the
    // actual returned nodes so the dashboard always matches the graph.
    const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    const countByType = {};
    nodes.forEach(n => {
        const type = String(n?.type || 'Custom');
        countByType[type] = (countByType[type] || 0) + 1;
    });

    const propertyCount = nodes.filter(n => {
        const type = String(n?.type || '').toLowerCase();
        return type === 'property' || type === 'blockchainland' || type.includes('land');
    }).length;

    const transactionCount = nodes.filter(n => {
        const type = String(n?.type || '').toLowerCase();
        return type.includes('transaction') || type === 'datasettransaction';
    }).length;

    const blockCount = nodes.filter(n => String(n?.type || '').toLowerCase() === 'block').length;

    document.getElementById('graph-count-nodes').textContent = nodes.length;
    document.getElementById('graph-count-edges').textContent = Array.isArray(graph.edges) ? graph.edges.length : 0;
    document.getElementById('graph-count-properties').textContent = propertyCount;
    document.getElementById('graph-count-transactions').textContent = transactionCount;
    document.getElementById('graph-count-blocks').textContent = blockCount;
    window.graphFocusNodeId = null;
    showGraphOverview();
}

function showGraphOverview() {
    if (!latestRealGraph || !latestRealGraph.nodes) return;

    // // Default view shows the mixed UK HM Land Registry + synthetic-owner dataset.
    // The initial payload is intentionally tiny so the graph stays readable; searches
    // can still open the matching history and include live blockchain records.
    const nodes = latestRealGraph.nodes;
    const ids = new Set(nodes.map(n => n.id));
    const edges = latestRealGraph.edges.filter(e => ids.has(e.source) && ids.has(e.target));
    renderGraphSubset(nodes, edges,
        `Readable overview: ${nodes.length} nodes from a small sample of recent dataset transactions. Search a User, Land, or Transaction to inspect related records.`);
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatGraphDate(node) {
    if (node?.transaction_date) return String(node.transaction_date);
    if (node?.date) return String(node.date);
    if (node?.timestamp) {
        const d = new Date(Number(node.timestamp) * 1000);
        if (!Number.isNaN(d.getTime())) return d.toLocaleDateString();
    }
    return 'Not available';
}

function renderGraphNodeDetails(found, allNodes, allEdges) {
    const box = document.getElementById('graph-selected-json');
    if (!box) return;

    const type = String(found.type || '').toLowerCase();
    const isTransaction = type.includes('transaction');
    const isProperty = type === 'property' || type === 'blockchainland';

    if (isTransaction) {
        const seller = found.old_owner || found.seller || 'Not available';
        const buyer = found.new_owner || found.buyer || 'Not available';
        const land = found.land_id || 'Not available';
        const year = found.year ?? 'Not available';
        const date = formatGraphDate(found);
        const price = Number(found.price ?? found.land_price ?? 0);
        const txid = found.transaction_id || (found.id || '').replace(/^DatasetTx::|^BlockchainTx::/, '');
        const label = found.label_name || (found.dataset_label === 1 ? 'Fraud / Duplicate' : 'Legitimate');

        box.innerHTML = `
            <div class="space-y-3">
                <div class="p-3 rounded-lg bg-indigo-50 border border-indigo-100">
                    <div class="font-bold text-indigo-900">
                        ${escapeHtml(seller)} <span class="text-indigo-500">→</span> ${escapeHtml(buyer)}
                    </div>
                    <div class="text-xs text-indigo-700 mt-1">
                        ${escapeHtml(seller)} transferred the land to ${escapeHtml(buyer)}
                        in <b>${escapeHtml(year)}</b>.
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-2">
                    <div><span class="text-[9px] uppercase font-bold text-slate-400">Transaction ID</span>
                        <div class="font-mono text-[10px] break-all">${escapeHtml(txid)}</div></div>
                    <div><span class="text-[9px] uppercase font-bold text-slate-400">Transfer Date</span>
                        <div>${escapeHtml(date)}</div></div>
                    <div><span class="text-[9px] uppercase font-bold text-slate-400">Land ID</span>
                        <div class="break-words">${escapeHtml(land)}</div></div>
                    <div><span class="text-[9px] uppercase font-bold text-slate-400">Price</span>
                        <div>₹${price.toLocaleString('en-IN')}</div></div>
                    <div><span class="text-[9px] uppercase font-bold text-slate-400">Status</span>
                        <div class="font-semibold">${escapeHtml(label)}</div></div>
                    <div><span class="text-[9px] uppercase font-bold text-slate-400">Source</span>
                        <div>${escapeHtml(found.source || '—')}</div></div>
                </div>

                ${found.document_hash ? `
                <div>
                    <span class="text-[9px] uppercase font-bold text-slate-400">Document Hash</span>
                    <div class="font-mono text-[10px] break-all">${escapeHtml(found.document_hash)}</div>
                </div>` : ''}
            </div>
        `;
        return;
    }
    // For an owner or land/property node, show the node plus the connected
    // transaction history so the ownership chain is immediately visible.
    const connectedIds = new Set();
    allEdges.forEach(e => {
        if (e.source === found.id) connectedIds.add(e.target);
        if (e.target === found.id) connectedIds.add(e.source);
    });

    const transactions = allNodes
        .filter(n => connectedIds.has(n.id) && String(n.type || '').toLowerCase().includes('transaction'))
        .sort((a, b) => Number(a.timestamp || 0) - Number(b.timestamp || 0));

    const historyHtml = transactions.length
        ? transactions.map(tx => {
            const seller = tx.old_owner || tx.seller || 'Unknown';
            const buyer = tx.new_owner || tx.buyer || 'Unknown';
            const year = tx.year ?? '—';
            return `<div class="py-2 border-b border-slate-100">
                <div class="font-semibold text-[11px]">${escapeHtml(seller)} → ${escapeHtml(buyer)}</div>
                <div class="text-[10px] text-slate-500">
                    ${escapeHtml(tx.land_id || '')} • ${escapeHtml(year)} •
                    ₹${Number(tx.price || 0).toLocaleString('en-IN')}
                </div>
            </div>`;
        }).join('')
        : '<div class="text-[10px] text-slate-400">No connected transaction history found.</div>';

    box.innerHTML = `
        <div class="space-y-3">
            <div><span class="text-[9px] uppercase font-bold text-slate-400">Type</span>
                <div class="font-semibold">${escapeHtml(found.type || 'Node')}</div></div>
            <div><span class="text-[9px] uppercase font-bold text-slate-400">Name</span>
                <div>${escapeHtml(found.label || found.id)}</div></div>
            ${found.land_id ? `<div><span class="text-[9px] uppercase font-bold text-slate-400">Land ID</span>
                <div class="break-words">${escapeHtml(found.land_id)}</div></div>` : ''}
            <div>
                <span class="text-[9px] uppercase font-bold text-slate-400">Transaction History</span>
                <div class="mt-1">${historyHtml}</div>
            </div>
        </div>
    `;
}

function renderConnectedNode(nodeId, depth = 2) {
    if (!latestRealGraph) return;
    const allNodes = latestRealGraph.nodes;
    const allEdges = latestRealGraph.edges;
    const ids = new Set([nodeId]);
    let frontier = [nodeId];
    for (let d = 0; d < depth; d++) {
        const next = [];
        const current = new Set(frontier);
        allEdges.forEach(e => {
            if (current.has(e.source) && !ids.has(e.target)) {
                ids.add(e.target); next.push(e.target);
            }
            if (current.has(e.target) && !ids.has(e.source)) {
                ids.add(e.source); next.push(e.source);
            }
        });
        frontier = next;
        if (!frontier.length) break;
    }

    const maxNodes = 80;
    let selectedIds = [...ids];
    if (selectedIds.length > maxNodes) {
        const direct = allEdges.filter(e => e.source === nodeId || e.target === nodeId);
        selectedIds = [...new Set([nodeId, ...direct.flatMap(e => [e.source, e.target])])].slice(0, maxNodes);
    }

    const selectedSet = new Set(selectedIds);
    const nodes = allNodes.filter(n => selectedSet.has(n.id));
    const edges = allEdges.filter(e => selectedSet.has(e.source) && selectedSet.has(e.target));
    const found = allNodes.find(n => n.id === nodeId);
    if (!found) return;

    window.graphFocusNodeId = nodeId;
    document.getElementById('graph-search').value = found.label || found.id;
    document.getElementById('graph-selected-node').classList.remove('hidden');
    renderGraphNodeDetails(found, allNodes, allEdges);

    renderGraphSubset(nodes, edges,
        `Focused on “${found.label || found.id}” • ${nodes.length} nodes • ${edges.length} edges • ${depth}-hop neighbourhood`);
}
async function focusGraphNode() {
    const query = document.getElementById('graph-search').value.trim();
    const status = document.getElementById('graph-search-status');
    const type = document.getElementById('graph-search-type').value;
    if (!query) {
        status.textContent = 'Enter a user name, land ID, or transaction ID.';
        return;
    }
    status.textContent = `Searching the full project transaction dataset for ${query}...`;
    await loadRealGraph({focusType:type, focusLabel:query, recentOnly:false});
}

async function loadRealGraph(options = {}) {
    const adminOnly = options.adminOnly === true;
    const recentOnly = options.recentOnly === true && !options.focusLabel && !adminOnly;
    const params = new URLSearchParams({
        limit: recentOnly ? '4' : (options.focusLabel ? '250' : '4'),
        // Keep the default overview focused on the dataset. Blockchain records
        // are included when a user searches, so the first render stays readable.
        include_blockchain: options.focusLabel || options.includeBlockchain ? '1' : '0',
        recent_only: recentOnly ? '1' : '0',
        admin_only: adminOnly ? '1' : '0'
    });
    const postcode = document.getElementById('graph-postcode').value.trim();
    const property = document.getElementById('graph-property').value.trim();
    if (postcode) params.set('postcode', postcode);
    if (property) params.set('property_id', property);
    if (options.focusType) params.set('focus_type', options.focusType);
    if (options.focusLabel) params.set('focus_label', options.focusLabel);

    try {
        const r = await fetch('/api/graph/real?' + params.toString());
        const j = await r.json();
        if (!j.success) {
            showMessage('Graph Error', j.message, true);
            return;
        }
        renderRealGraph(j);
        if (options.focusLabel) {
            const wantedType = String(options.focusType || '').toLowerCase();
            const target = (options.preferCustom ? j.nodes.find(n => String(n.source || '').toLowerCase() === 'user-added' && String(n.label || '').toLowerCase() === String(options.focusLabel).toLowerCase()) : null) || j.nodes.find(n => {
                const type = String(n.type || '').toLowerCase();
                const label = String(n.label || '').toLowerCase();
                const wanted = String(options.focusLabel).toLowerCase();
                const nodeId = String(n.id || '').toLowerCase();
                const transactionId = String(n.transaction_id || '').toLowerCase();
                const landId = String(n.land_id || '').toLowerCase();

                return (wantedType === 'owner' && type === 'owner' && (label === wanted || nodeId === `owner::${wanted}`)) ||
                       (wantedType === 'property' && (type === 'property' || type === 'blockchainland') &&
                           (label === wanted || landId === wanted || nodeId === `property::${wanted}` || nodeId === `blockchainland::${wanted}`)) ||
                       (wantedType === 'transaction' && type.includes('transaction') &&
                           (label === wanted ||
                            transactionId === wanted ||
                            nodeId === wanted ||
                            nodeId.endsWith(`::${wanted}`) ||
                            transactionId.endsWith(wanted) ||
                            label.endsWith(wanted)));
            });
            if (target) {
                const depth = parseInt(document.getElementById('graph-depth').value || '1', 10);
                renderConnectedNode(target.id, depth);
            }
            document.getElementById('graph-search-status').textContent = target
                ? `Showing ${options.focusType === 'Owner' ? 'User' : options.focusType} ${options.focusLabel} and its related transactions.`
                : `No ${options.focusType || 'node'} named ${options.focusLabel} was found in the project transaction dataset.`;
        } else if (adminOnly) {
            document.getElementById('graph-search-status').textContent = 'Network/admin graph. Use only when you want to inspect blockchain network nodes.';
        } else {
            showMessage('Graph Updated', recentOnly
                ? 'Showing only recently added blockchain/custom nodes.'
                : 'Showing a compact source graph. Search a User, Land, or Transaction for detailed connections.');
        }
    } catch (err) {
        document.getElementById('graph-search-status').textContent = 'Graph loading failed: ' + err.message;
    }
}

async function addGraphNode() {
    const type = document.getElementById('custom-node-type').value.trim() || 'Custom';
    const label = document.getElementById('custom-node-label').value.trim();
    if (!label) {
        showMessage('Graph Error', 'Enter a node label.', true);
        return;
    }
    const r = await fetch('/api/graph/add_node', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({type, label})
    });
    const j = await r.json();
    showMessage(j.success ? 'Graph Node Added' : 'Graph Error', j.message, !j.success);
    if (j.success) {
        // Immediately switch to the new node's neighbourhood. The backend
        // resolves every matching historical transaction for Owner/Land/Transaction.
        await loadRealGraph({focusType:type, focusLabel:label, recentOnly:false, preferCustom:true});
        if (j.node) {
            window.graphFocusNodeId = j.node.id;
        }
    }
}

async function loadConsensusExplanation() {
    const selectedEl = document.getElementById('xai-consensus-selected');
    if (!selectedEl) return;
    try {
        const response = await fetch('/api/consensus/explanation', {headers: {'Accept': 'application/json'}});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || data.error || 'Request failed');
        if (data.error && !data.selected_consensus) {
            selectedEl.textContent = 'Model not ready';
            document.getElementById('xai-consensus-reason').textContent = data.error;
            document.getElementById('xai-lime-target').textContent = 'LIME not run — no captured dynamic-consensus selection.';
            document.getElementById('xai-lime-features').innerHTML = '';
            document.getElementById('xai-consensus-note').textContent = 'Submit a transaction to trigger the existing dynamic-consensus selector, then refresh.';
            return;
        }
        selectedEl.textContent = data.selected_consensus || '—';
        document.getElementById('xai-consensus-reason').textContent = data.reason || 'No reason available.';
        document.getElementById('xai-current-consensus').textContent = data.current_consensus || '—';
        document.getElementById('xai-current-status').textContent = data.selection_matches_current
            ? 'The captured selection matches the active consensus mode.'
            : `The captured selection and active protocol differ; check whether another authorized action changed the active mode.`;
        document.getElementById('xai-risk').textContent = `${Number(data.risk_score || 0).toFixed(1)}%`;
        document.getElementById('xai-trust').textContent = `${Number(data.trust_score || 0).toFixed(1)}/100`;
        const net = data.network || {};
        document.getElementById('xai-network-summary').textContent = `${net.active_nodes ?? '—'} application nodes · latency ${net.latency_ms ?? 'Unavailable'} ms · throughput ${net.throughput_tps ?? 'Unavailable'} TPS`;
        const sourceMap = data.input_sources || {};
        document.getElementById('xai-input-sources').textContent = 'Input sources: ' + Object.entries(sourceMap).map(([k, v]) => `${k}: ${v}`).join(' | ');
        const factors = Array.isArray(data.factors) ? data.factors : [];
        document.getElementById('xai-consensus-factors').innerHTML = factors.map(item => `
            <div class="rounded-lg border border-slate-200 bg-white p-3">
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1"><strong class="text-sm text-slate-800">${escapeConsensusText(item.factor)}</strong><span class="text-sm font-bold text-slate-900">${escapeConsensusText(item.value)}</span></div>
                <p class="text-sm text-slate-600 mt-1.5 leading-relaxed">${escapeConsensusText(item.effect)}</p>
            </div>`).join('') || '<div class="text-sm text-slate-500">No decision factors available.</div>';
        document.getElementById('xai-consensus-algorithm').textContent = data.algorithm_explanation || '';
        document.getElementById('xai-consensus-note').textContent = data.method_note || '';
        const lime = data.lime_explanation || {};
        document.getElementById('xai-lime-target').textContent = `${lime.method || 'LIME'} · Explained class: ${lime.target_consensus || data.selected_consensus || '—'}`;
        const limeFeatures = Array.isArray(lime.features) ? lime.features : [];
        const maxWeight = Math.max(0.0001, ...limeFeatures.map(item => Math.abs(Number(item.weight || 0))));
        document.getElementById('xai-lime-features').innerHTML = limeFeatures.map(item => {
            const weight = Number(item.weight || 0);
            const width = Math.max(2, Math.min(100, Math.abs(weight) / maxWeight * 100));
            const supports = weight >= 0;
            return `<div class="rounded-lg border border-violet-100 bg-white p-3">
                <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1"><strong class="text-sm text-slate-800">${escapeConsensusText(item.feature)}</strong><span class="text-sm font-bold ${supports ? 'text-emerald-700' : 'text-rose-700'}">${weight > 0 ? '+' : ''}${weight.toFixed(4)}</span></div>
                <div class="h-2 bg-slate-100 rounded-full mt-2 overflow-hidden"><div class="h-full rounded-full ${supports ? 'bg-emerald-500' : 'bg-rose-500'}" style="width:${width}%"></div></div>
                <div class="text-xs text-slate-500 mt-1.5">${escapeConsensusText(item.direction || (supports ? 'Supports selected class locally' : 'Opposes selected class locally'))}</div>
            </div>`;
        }).join('') || '<div class="text-sm text-slate-500 bg-white rounded-lg p-3">LIME explanation unavailable. Check the Flask terminal for the underlying error.</div>';
        document.getElementById('xai-lime-warning').textContent = lime.warning || lime.scope || '';
        const options = data.algorithm_options || {};
        const descriptions = {
          'PoW': ['Proof of Work', 'Nodes compete through computational work. Strong historical use in public chains, but high energy and slower finality can be drawbacks.'],
          'PoS': ['Proof of Stake', 'Validators are selected based on stake and protocol rules. Usually more energy-efficient than PoW; requires secure staking and slashing/governance design.'],
          'PBFT': ['Practical Byzantine Fault Tolerance', 'Known validators exchange messages/votes and tolerate some Byzantine faults. Fast finality in small permissioned groups, but communication overhead grows with validator count.'],
          'Raft': ['Raft', 'Leader-based crash-fault-tolerant consensus. Simple and efficient for trusted nodes, but it is not designed to tolerate malicious Byzantine validators.'],
          'HotStuff': ['HotStuff', 'Quorum-based Byzantine-fault-tolerant protocol with chained phases. Designed for scalable BFT-style consensus, but implementation and validator assumptions matter.']
        };
        const actualSelection = data.selected_consensus;
        document.getElementById('xai-algorithm-options').innerHTML = (data.algorithms || Object.keys(descriptions)).map(name => {
          const info = descriptions[name] || [name, 'Protocol details are not available.'];
          const chosen = name === actualSelection;
          const policyReason = options[name] || (chosen ? data.reason : 'This protocol was not selected in the captured decision.');
          return `<article class="rounded-xl border ${chosen ? 'border-emerald-300 bg-emerald-50/60' : 'border-slate-200 bg-white'} p-4">
            <div class="flex items-start justify-between gap-2"><div><h4 class="font-bold text-slate-900">${escapeConsensusText(name)}</h4><p class="text-xs text-slate-500 mt-0.5">${escapeConsensusText(info[0])}</p></div>${chosen ? '<span class="shrink-0 rounded-full bg-emerald-100 text-emerald-800 text-[10px] font-bold px-2 py-1">Actually selected</span>' : ''}</div>
            <p class="text-sm text-slate-600 leading-relaxed mt-3">${escapeConsensusText(info[1])}</p>
            <div class="mt-3 border-t border-slate-100 pt-3"><div class="text-[10px] uppercase tracking-wide font-bold text-slate-500">Role in this decision</div><p class="text-sm text-slate-700 mt-1 leading-relaxed">${escapeConsensusText(policyReason)}</p></div>
          </article>`;
        }).join('');
    } catch (error) {
        selectedEl.textContent = 'Explanation unavailable';
        document.getElementById('xai-consensus-reason').textContent = error.message;
    }
}
function escapeConsensusText(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

async function refresh() {
    try {
        const r = await fetch('/api/state');
        currentState = await r.json();
        const rUsers = await fetch('/api/get_users');
        currentUsers = await rUsers.json();

        document.getElementById('current-user').textContent = currentState.current_user;
        document.getElementById('current-consensus').innerHTML = getConsensusTag(currentState.consensus_mode);
        document.getElementById('predicted-consensus').textContent = currentState.predicted_mode || 'Analyzing...';
        document.getElementById('difficulty').textContent = currentState.difficulty;
        document.getElementById('pending-tx-count').textContent = currentState.pending_transactions.length;

        renderMLStatus(currentState.ml_ready);
        renderNodeList(currentState.nodes);
        
        const selectEl = document.getElementById('consensus');
        const currentMode = currentState.consensus_mode;
        selectEl.innerHTML = currentState.consensus_algos.map(mode =>
            `<option value="${mode}" ${mode === currentMode ? 'selected' : ''}>${mode}</option>`
        ).join('');

        renderChain(currentState.chain, currentState.difficulty);
        renderPendingTx(currentState.pending_transactions);
        renderBlockedTx(currentState.blocked_transactions || []);
        renderGovernance(currentUsers);
        loadConsensusExplanation();
    } catch (e) { console.error("Refresh failed", e); }
}

async function addNode() {
    const stake = document.getElementById('new-stake').value;
    const r = await fetch('/api/add_node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stake: parseInt(stake) || 100 })
    });
    const j = await r.json();
    showMessage(j.success ? 'Success' : 'Error', j.message, !j.success);
    refresh();
}

async function removeNode(name) {
    const r = await fetch('/api/remove_node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    });
    const j = await r.json();
    showMessage(j.success ? 'Node Removed' : 'Error', j.message, !j.success);
    refresh();
}

async function addLandTx() {
    const landId = document.getElementById('land-id').value;
    const newOwner = document.getElementById('new-owner').value;
    const landPrice = document.getElementById('land-price').value;
    const pdf = document.getElementById('land-pdf').files[0];

    if (!landId || !newOwner || !landPrice || !pdf) {
        showMessage("Validation Error", "All fields and PDF are required", true);
        return;
    }

    const formData = new FormData();
    formData.append("land_id", landId);
    formData.append("new_owner", newOwner);
    formData.append("land_price", landPrice);
    formData.append("pdf", pdf);

    const r = await fetch('/api/add_land_tx', { method: 'POST', body: formData });
    const j = await r.json();

    showMessage(
        j.success ? "Transaction Queued" : (j.blocked ? "AI Blocked Transaction" : "Error"),
        j.message,
        !j.success
    );

    // Keep Srija's original GNN + SHAP + blocking workflow untouched.
    // Only add graph visualisation/search as an additional view.
    if (j.transaction && (j.success || j.blocked)) {
        await refresh();

        const txKey = j.transaction.id || j.transaction.transaction_id || j.transaction.land_id;
        const txType = (j.transaction.id || j.transaction.transaction_id) ? 'Transaction' : 'Property';
        await loadRealGraph({
            focusType: txType,
            focusLabel: txKey,
            recentOnly: false
        });

        document.getElementById('graph-search-status').textContent = j.blocked
            ? `Showing the blocked transaction: ${j.transaction.old_owner || currentState.current_user} → ${j.transaction.new_owner || newOwner}.`
            : `Showing the new transfer: ${j.transaction.old_owner || currentState.current_user} → ${j.transaction.new_owner || newOwner}.`;
        return;
    }

    refresh();
}

async function triggerConsensus() {
    const mode = document.getElementById('consensus').value;
    await fetch('/api/set_consensus', {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ mode })
    });
    const r = await fetch('/api/trigger', { method: 'POST' });
    const j = await r.json();
    showMessage(j.success ? 'Block Forged' : 'Forging Failed', j.result, !j.success);
    refresh();
    loadRealGraph({recentOnly:false});
}

async function resolve() {
    const r = await fetch('/api/resolve', { method: 'POST' });
    const j = await r.json();
    showMessage('Network Sync', j.message);
    refresh();
    loadRealGraph({recentOnly:false});
}

async function voteUser(username) {
    const r = await fetch('/api/vote_user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
    });
    const j = await r.json();
    showMessage(j.success ? 'Vote Cast' : 'Error', j.message, !j.success);
    refresh();
}

document.getElementById('consensus').addEventListener('change', async function(){
    const mode = this.value;
    await fetch('/api/set_consensus', {
        method:'POST', 
        headers:{'content-type':'application/json'}, 
        body: JSON.stringify({mode})
    });
    showMessage('Policy Updated', `Consensus protocol changed to ${mode}`);
    refresh();
});

refresh();
// Load the user-facing mixed dataset graph immediately.
// Admin/network nodes are available only through the dedicated admin API mode.
loadRealGraph({recentOnly:true});
setInterval(refresh, 4000);
</script>

</body>
</html>
"""
if __name__ == '__main__':
    # Add a couple of initial nodes for a better start
    node_names = ['Node-B', 'Node-C']
    for name in node_names:
        network.add_node(name, stake=200 + random.randint(0, 100))
    print("\n--- ConcordiaChain Simulator ---")
    print(f"Credentials loaded from: {CREDENTIALS_FILE}")
    print(f"Total Users: {len(USERS)}")
    app.run(debug=True, use_reloader=False, threaded=False)