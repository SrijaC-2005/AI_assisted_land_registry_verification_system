
# Dynamic-Consensus-Blockchain-for-Secure-Digital-Land-Registry-using-AI-PQC

Welcome to this **next-generation digital land registry platform** — a secure, adaptive, and community-governed system built on blockchain technology.

This project simulates how land ownership can be recorded, verified, and transferred using modern techniques such as:

* **Machine learning to dynamically choose consensus methods**
* **Post-quantum cryptography for future-proof digital signatures**
* **Decentralized governance where peers approve new users**
* **Tamper-proof, immutable records of land transactions**

This combination of technologies makes the system **secure, transparent, resilient, and adaptable to changing network conditions**.

---

## What It Does

Traditional land registry systems are often slow, paper-based, and vulnerable to tampering or fraud. This project demonstrates an innovative approach by:

* Letting users upload land deed PDFs where the system computes a **secure hash** for verification.
* Recording ownership transfers as **blockchain transactions**.
* Using **Post-Quantum Cryptographic (PQC) signatures** to make records safe even against future quantum attacks.
* Employing an **ML model** to predict the most efficient consensus protocol (like PoW, PoS, PBFT, Raft, or HotStuff) based on current network behavior and load.
* Including a **peer-review system** where existing users vote to approve or reject new participants.

This makes the platform suitable as a **proof-of-concept for modern, secure, and decentralized land records** that governments, municipalities, or smart cities could adopt.

---

## Features At a Glance

### Core Functionality
**User Management**
- Secure account registration with hashed passwords (scrypt)
- Community-based approval system with voting
- Founder users with immediate active status
- Role-based access control

**Land Transaction Management**
- Upload land deed PDFs with automatic hash verification
- PQC digital signatures for transaction integrity
- Transaction queuing system
- Real-time transaction queue display

**Blockchain & Consensus**
- Multi-algorithm consensus support (PoW, PoS, PBFT, Raft, HotStuff)
- ML-based dynamic consensus selection (when models available)
- Immutable ledger with cryptographic verification
- Multi-node blockchain simulator

**Governance & Transparency**
- Decentralized peer-review system
- Live governance dashboard showing pending approvals
- Voting progress tracker
- Transparent ledger display with full transaction details

**Data Persistence**
- Automatic file saving (CSV + JSON)
- State recovery on application restart
- No hardcoded paths — fully portable

**User Interface**
- Modern, responsive design with Tailwind CSS
- Real-time dashboard updates
- Clear error messages and status indicators
- Professional land registry interface

---

## Why It Matters

By blending **AI with blockchain and advanced cryptographic techniques**, this project goes beyond static systems. It simulates a world where:

* The blockchain adapts intelligently to conditions
* Security remains strong even in a post-quantum era
* Users collectively govern system access
* Land ownership becomes transparent and tamper-resistant

This isn’t just code — it’s a **vision for future digital infrastructure**.

---

## Tech Stack

* **Python & Flask** — backend and web server
* **Machine Learning** — dynamic consensus prediction
* **Post-Quantum Cryptography (PQC)** — secure signature generation
* **JSON + CSV** — persistent storage
* **Tailwind CSS + HTML** — clean user interface

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- All required dependencies installed

### Installation Steps

1. **Clone or navigate to the project directory**

   ```bash
   cd "Dynamic-Consensus-Blockchain-for-Secure-Digital-Land-Registry-using-AI-PQC"
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Required packages: Flask, pandas, numpy, joblib, quantcrypt, werkzeug

3. **Project Structure**
   
   **Important**: The application uses **relative paths** from the project root directory. Ensure all files are in their proper locations:

   ```
   project-root/
   ├── app.py                          (Main Flask application)
   ├── User_Credential.csv             (User credentials - auto-created)
   ├── Blockchain_Ledger.json          (Blockchain state - auto-created)
   ├── Dataset/
   │   ├── Current_Network_Metrics.csv
   │   └── Dataset_Generator.py
   ├── FL_SGD/
   │   ├── federated_sgd_global_model.pkl (optional)
   │   ├── fed_label_encoder.pkl          (optional)
   │   └── fed_scaler.pkl                 (optional)
   └── DecisionTree/, FL_Random_Forest/   (other ML modules)
   ```

   **Good**: Run from project root directory
   ```bash
   cd "D:\Project Work - II\Dynamic-Consensus-Blockchain-for-Secure-Digital-Land-Registry-using-AI-PQC"
   python app.py
   ```

   **Bad**: Don't run from a different directory or with hardcoded paths

4. **Run the application**

   ```bash
   python app.py
   ```

   The app will:
   - Automatically create `User_Credential.csv` if it doesn't exist
   - Automatically create `Blockchain_Ledger.json` for blockchain persistence
   - Initialize default founder users (Sengathir, UserB, vishal)
   - Start Flask server on `http://127.0.0.1:5000`

5. **Access the application**

   Open your browser and go to:
   ```
   http://localhost:5000
   ```

---

### First-Time Setup

When you run the app for the first time:

1. **Founder Users** are automatically created and marked as **ACTIVE**:
   - **Username**: Sengathir | **Password**: 12245
   - **Username**: UserB | **Password**: defaultpass
   - **Username**: vishal | **Password**: vishal

2. **No additional setup required** — the system initializes blockchain and user database automatically.

3. **Files Created**:
   - `User_Credential.csv` — Stores all user credentials (hashed passwords)
   - `Blockchain_Ledger.json` — Stores all blockchain blocks and transactions

---

### Important Notes

**Path Configuration**:
- All file paths are now **relative to the project root directory**
- NO hardcoded paths (like `D:\PW_II\Review 1\`)
- App works correctly when run from the project directory
- Data persists between app restarts in same directory

**ML Models** (Optional):
- If ML model files are missing, app runs in **static consensus mode**
- This is normal — core functionality works perfectly without ML
- To enable dynamic ML-based consensus, place model files in `FL_SGD/` directory:
  - `federated_sgd_global_model.pkl`
  - `fed_label_encoder.pkl`
  - `fed_scaler.pkl`

---

## How It Works

### User Registration & Community Approval

1. **Founder Users** are automatically active on startup (no approval needed):
   - Can immediately add transactions and forge blocks
   - Can approve new users by voting

2. **New Users** sign up and their account is pending approval:
   - Status: "Account pending COMMUNITY APPROVAL"
   - Existing active users can see pending users in "Network Governance"
   - Majority vote (>50%) activates the new user

3. **Once Approved**, new users become active and can:
   - Submit land transactions
   - Forge blocks with chosen consensus algorithm
   - Vote on future user approvals

### Transaction & Blockchain Workflow

1. **Active user submits land transaction** with PDF document:
   - System computes secure hash of the document
   - Post-quantum signature is generated
   - Transaction added to pending queue

2. **Transaction appears in live queue** on the dashboard

3. **User triggers consensus** to forge the current block:
   - Selects consensus algorithm (PoW, PoS, PBFT, Raft, or HotStuff)
   - Clicks "Forge Current Block"
   - ML model predicts optimal algorithm (if available)

4. **Block is created and added to immutable ledger**:
   - Contains all pending transactions
   - Includes cryptographic proofs
   - Permanently stored in `Blockchain_Ledger.json`

### Data Persistence

All data is automatically saved:
- **User Credentials** → `User_Credential.csv` (passwords hashed with scrypt)
- **Blockchain State** → `Blockchain_Ledger.json` (all blocks and transactions)
- **Reload on restart** → System automatically loads previous state

---

## Governance & Community

Instead of a central administrator, this platform uses a **peer-review governance model** where existing participants:

* Vote to approve new users
* Participate in consensus decisions
* Help maintain network integrity

This reflects real decentralized decision-making.

---

## API Endpoints (For Reference)

### Authentication
- `GET /login` — Login page
- `POST /login` — Submit login credentials
- `GET /signup` — Sign up page
- `POST /signup` — Register new account
- `GET /logout` — Logout user

### Blockchain Operations
- `POST /api/add_land_tx` — Submit land transaction
- `POST /api/trigger` — Forge new block (trigger consensus)
- `POST /api/set_consensus` — Change consensus algorithm
- `GET /api/state` — Get current blockchain state
- `GET /api/get_users` — Get active and pending users

### Governance
- `POST /api/vote_user` — Vote to approve pending user
- `POST /api/add_node` — Add blockchain node
- `POST /api/remove_node` — Remove blockchain node

---

## Use Cases

* Proof of concept for **government land registries**
* Testing **adaptive consensus algorithms**
* Research on **blockchain + AI + PQC integration**
* Learning platform for **decentralized systems architecture**

---

## License

This project is open-source and free to use under the [MIT License](LICENSE).

---
