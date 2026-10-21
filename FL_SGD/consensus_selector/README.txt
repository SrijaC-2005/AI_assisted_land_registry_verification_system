# Land Registry — corrected integration files

## What is corrected
- `app.py`: falls back to the supplied trained 8-feature consensus-selector artifacts and LIME background when the `FL_SGD` paths or `Dataset/Current_Network_Metrics.csv` are absent; selects a deterministic latest complete feature row rather than a random row; records provenance as dataset/simulated unless real telemetry is instrumented; refuses unsupported model labels; and displays missing risk/trust values as `Unavailable`.
- `consensus_explainer.py`: LIME explains the same trained model and input row; missing LIME/degraded background data no longer hides the model prediction or changes consensus; it reports a clear warning instead.
- `train_consensus_selector.py`: accepts either supported labelled metrics CSV filename and explains why retraining cannot use `lime_background.csv` by itself.
- `real_dataset_graph.py`: included unchanged because the uploaded version's graph builder is syntactically valid and the `_node` calls do not pass a duplicate `label` argument.
- `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `lime_background.csv`, `evaluation.json`: the uploaded consensus-selector artifacts and explanation background.

## Install into your existing project
1. Back up your current project folder first.
2. Copy `app.py`, `consensus_explainer.py`, and `train_consensus_selector.py` into the existing project root, replacing those files.
3. Replace `real_dataset_graph.py` only if you need the included same-version copy; it was not functionally changed.
4. Keep the `risk_assessment/` package, templates/data folders, and all other existing project files. This ZIP is a patch bundle, not a complete standalone application.
5. Ensure dependencies are installed in the same virtual environment used to run Flask: `pip install lime pandas numpy scikit-learn joblib`
6. Run the app as you normally do. After a successful land-transfer request, refresh Module 4 to view the captured decision.

## Important limitations
- The uploaded `lime_background.csv` has the eight feature columns but no `consensus` target; it can support LIME and inference, but cannot retrain the selector.
- The bundled pickle artifacts were created with a different scikit-learn version than the current environment may use. If loading fails or produces a compatibility warning, retrain in the project's actual environment from a labelled metrics CSV. Do not treat the saved `evaluation.json` accuracy as proof of real-world consensus optimality.
- The fallback selector input comes from project background data, not live network telemetry. Wire actual instrumentation before describing the result as live/dynamic network measurement.
- This patch does not repair or replace the GNN fraud-risk training pipeline or invent missing `Dataset/land_transactions.csv` / `real_augmented_transactions.json` files.
- Do not replace `Blockchain_Ledger.json` with the uploaded `Blockchain_Ledger_corrupted_backup(2).json`; that backup was not included or altered by this patch.
