from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
import time, json, hashlib, random
from typing import List, Dict, Any
import os, uuid, itertools
import csv
from numpy import block
from numpy import block
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib
import base64

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

NETWORK_STATE_FILE = os.path.join(BASE_DIR, "Blockchain_Ledger.json") # New file for state persistence
USERS: Dict[str, str] = {} # {username: hashed_password}
MAX_NODES = 5


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
        seller = tx["old_owner"]
        buyer = tx["new_owner"]

        # -------------------------
        # Check Pending Queue
        # -------------------------
        for pending in self.pending_transactions:

            if pending["land_id"] == land_id:
                return False, "Land ID already exists in Pending Queue."

            if (pending["old_owner"] == seller and
                pending["new_owner"] == buyer and
                pending["land_id"] == land_id):
                return False, "Same buyer/seller transaction already exists in Pending Queue."


        # -------------------------
        # Check Blockchain Ledger
        # -------------------------
        for block in self.chain:
            print(type(block))
            print(block)
            print(block.__dict__)
            print("Checking Block:", block.index)

            for ledger_tx in block.transactions:

                print("Ledger Land ID =", ledger_tx.get("land_id"))
                print("Incoming Land ID =", land_id)

                if "land_id" not in ledger_tx:
                    continue

                if ledger_tx["land_id"] == land_id:
                    print("MATCH FOUND")
                    return False, "Land ID already exists in Ledger."

                if (
                    ledger_tx["old_owner"] == seller
                    and ledger_tx["new_owner"] == buyer
                    and ledger_tx["land_id"] == land_id
                ):
                    return False, "Same buyer/seller transaction already exists in Ledger."

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
    "timestamp": time.time(),
    "id": str(uuid.uuid4())
}
        ok, message = self.check_duplicate_transaction(tx)

        if not ok:
            return False, message
        self.pending_transactions.append(tx)
        return True, tx


    def set_consensus(self, mode):
        if mode in self.consensus_algos:
            self.consensus_mode = mode
            return True
        return False

    def run_pow(self, proposer_node: str) -> Block:
        last_block = self.get_last_block()
        new_block = Block(
            index=last_block.index + 1,
            timestamp=time.time(),
            transactions=[self.pending_transactions[0]],
            previous_hash=last_block.hash(),
            proposer=proposer_node
        )
        if new_block.transactions:
             new_block.transactions[0]['predicted_consensus'] = 'PoW'
        while True:
            new_block.nonce += 1
            hash_attempt = new_block.hash()
            if hash_attempt.startswith('0' * self.difficulty):
                self.pending_transactions.pop(0)
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
    """Serializes and saves the network object's state to disk."""
    global network
    try:
        # Simplify the network state for JSON serialization
        state = {
            'chain': [block.to_dict() for block in network.chain],
            'pending_transactions': network.pending_transactions,
            'nodes': network.nodes,
            'consensus_mode': network.consensus_mode,
            'difficulty': network.difficulty,
            'consensus_algos': network.consensus_algos,
            'active_users': network.active_users,
            'pending_users': network.pending_users,
        }
        with open(NETWORK_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        # print("💾 Network state saved.") # Disabled for cleaner console
    except Exception as e:
        print(f"❌ Error saving network state: {e}")

def load_network_state():
    """Loads and deserializes the network object's state or initializes default."""
    global network
    network = Network() # Initialize a fresh network object first
    
    if os.path.exists(NETWORK_STATE_FILE):
        try:
            with open(NETWORK_STATE_FILE, 'r') as f:
                state = json.load(f)
            
            # Rebuild chain (must recreate Block objects from dicts)
            network.chain = []
            for block_data in state.get('chain', []):
                block = Block(
                    index=block_data['index'],
                    timestamp=block_data['timestamp'],
                    transactions=block_data['transactions'],
                    previous_hash=block_data['previous_hash'],
                    nonce=block_data.get('nonce', 0),
                    proposer=block_data.get('proposer')
                )
                network.chain.append(block)

            # Restore simple properties
            network.pending_transactions = state.get('pending_transactions', [])
            network.nodes = state.get('nodes', network.nodes) # Keep Node-A if state is missing nodes
            network.consensus_mode = state.get('consensus_mode', 'PoW')
            network.difficulty = state.get('difficulty', 3)
            network.active_users = state.get('active_users', {})
            network.pending_users = state.get('pending_users', {})
            
            # Re-ensure genesis block if chain was empty or failed to load
            if not network.chain:
                 network.create_genesis_block()

            # print("✅ Network state loaded from file.") # Disabled for cleaner console
            
        except Exception as e:
            print(f"❌ Error loading network state: {e}. Reinitializing network.")
            network = Network() # Reset on failure

    # Always ensure default users are set up
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
        response = func(*args, **kwargs)
        save_network_state()
        return response
    wrapper.__name__ = func.__name__ + '_persisted'
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
@persist_action
def add_land_tx():
    current_user = session.get('username', 'Anonymous')
    
    # Check if user is active (approved by community)
    load_network_state()
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
        return jsonify({'success': False, 'message': result})

    # ---- ML Consensus Prediction (UNCHANGED) ----
    predicted_consensus = network.consensus_mode
    if model_ready:
        try:
            random_row = df_traffic.sample(n=1)

            # features
            features_df = random_row[MODEL_FEATURES]
            scaled_features = scaler.transform(features_df.values)

            # prediction
            prediction = model.predict(scaled_features)[0]
            predicted_consensus = label_encoder.inverse_transform([prediction])[0]

            # TRUE LABEL from dataset
            true_label = random_row['consensus'].values[0]

            correct = predicted_consensus == true_label
            print(f"Prediction: {predicted_consensus} | True: {true_label} | Correct: {correct}")

            # online supervised learning
            online_update(scaled_features, true_label)

            network.set_consensus(predicted_consensus)

        except Exception as e:
            print("ML error:", e)


    result["predicted_consensus"] = predicted_consensus

    return jsonify({
        'success': True,
        'message': 'Land transaction added to queue.',
        'transaction': result
    })


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

    <div class="card p-6 mb-8">
        <h2 class="text-sm font-bold mb-4 text-slate-800 flex items-center gap-2">
            <span class="w-2 h-5 bg-blue-600 rounded-full"></span>
            Live Transaction Queue
        </h2>

        <div id="pending-tx-view" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
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
        pendingTxViewEl.innerHTML = '<div class="col-span-full py-8 text-center text-slate-400 text-sm italic text-sm">No transactions currently in queue...</div>';
        return;
    }

    txList.forEach(tx => {
        pendingTxViewEl.innerHTML += `
            <div class="p-4 rounded-lg bg-white border border-slate-200 shadow-sm">
                <div class="flex justify-between items-start mb-2">
                    <span class="text-blue-600 font-mono text-[10px]">TX_${formatHash(tx.id)}</span>
                    <span class="text-slate-800 font-bold text-xs"">₹${tx.land_price.toLocaleString("en-IN")}</span>
                </div>
                <div class="text-[11px] space-y-1">
                    <p><span class="text-slate-500">FROM:</span> ${tx.old_owner}</p>
                    <p><span class="text-slate-500">TO:</span> ${tx.new_owner}</p>
                </div>
            </div>
        `;
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
        renderGovernance(currentUsers);
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
    showMessage(j.success ? "Transaction Queued" : "Error", j.message, !j.success);
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
}

async function resolve() {
    const r = await fetch('/api/resolve', { method: 'POST' });
    const j = await r.json();
    showMessage('Network Sync', j.message);
    refresh();
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
    app.run(debug=True, use_reloader=False)
