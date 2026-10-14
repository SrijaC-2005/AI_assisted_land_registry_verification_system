"""Interactive graph data service for Srija's land-registry project.

This module is deliberately separate from risk_assessment/graph_builder.py.

- risk_assessment/graph_builder.py builds the graph used by the GNN.
- this module builds JSON for the browser's interactive graph/search UI.

Data source used for the dataset layer:
    real_augmented_transactions.json

This is the project's mixed dataset: UK HM Land Registry Price Paid Data
augmented with synthetic owner/buyer/seller details because the public PPD
dataset does not provide party names. The graph does NOT generate new
transactions or new synthetic owners; it only visualizes the supplied mixed
dataset.

Additional live sources:
    Blockchain_Ledger.json (through app.py's live network state)
    Graph_Nodes.json (created automatically for user-added graph nodes)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional


class RealDatasetGraph:
    def __init__(self, dataset_path: str, custom_store: str):
        self.dataset_path = dataset_path
        self.custom_store = custom_store
        self._cache_data: Optional[List[Dict[str, Any]]] = None
        self._cache_mtime: Optional[float] = None
        self.custom_graph = self._load_custom_graph()

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def _load_custom_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(self.custom_store):
            return {"nodes": [], "edges": []}
        try:
            with open(self.custom_store, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "nodes": list(data.get("nodes", [])),
                "edges": list(data.get("edges", [])),
            }
        except Exception as exc:
            print(f"Graph custom-store load warning: {exc}")
            return {"nodes": [], "edges": []}

    def _save_custom_graph(self) -> None:
        os.makedirs(os.path.dirname(self.custom_store) or ".", exist_ok=True)
        with open(self.custom_store, "w", encoding="utf-8") as f:
            json.dump(self.custom_graph, f, indent=2, default=str)

    def _load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            return []

        mtime = os.path.getmtime(self.dataset_path)
        if self._cache_data is not None and self._cache_mtime == mtime:
            return self._cache_data

        rows: List[Dict[str, Any]] = []
        try:
            if self.dataset_path.lower().endswith(".json"):
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Mixed dataset JSON must contain a list of transactions")
                rows = [dict(row) for row in data if isinstance(row, dict)]
            else:
                with open(self.dataset_path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = [dict(row) for row in reader]

            self._cache_data = rows
            self._cache_mtime = mtime
            return rows
        except Exception as exc:
            print(f"Graph dataset load error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_price(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _node(node_id: str, node_type: str, label: str, **extra: Any) -> Dict[str, Any]:
        return {"id": node_id, "type": node_type, "label": label, **extra}

    @staticmethod
    def _property_id(land_id: str) -> str:
        raw = str(land_id or "").strip()
        return "Property::" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _transaction_id(row: Dict[str, Any]) -> str:
        value = str(
            row.get("transaction_id")
            or row.get("id")
            or ""
        ).strip()
        if value:
            return value
        raw = json.dumps(row, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _timestamp_sort_value(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    @classmethod
    def _date_label(cls, value: Any) -> str:
        if value in (None, ""):
            return ""
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            try:
                return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d")
            except Exception:
                return str(value)

    @classmethod
    def _year_label(cls, value: Any) -> Optional[int]:
        date = cls._date_label(value)
        try:
            return int(date[:4])
        except Exception:
            return None

    @staticmethod
    def _matches_focus(row: Dict[str, Any], focus_type: str, focus_label: str) -> bool:
        wanted = str(focus_label or "").strip().casefold()
        if not wanted:
            return True

        tx_id = str(row.get("transaction_id") or row.get("id") or "").strip().casefold()
        land_id = str(row.get("land_id") or "").strip().casefold()
        seller = str(row.get("old_owner") or "").strip().casefold()
        buyer = str(row.get("new_owner") or "").strip().casefold()
        focus = str(focus_type or "").strip().casefold()

        if focus in {"owner", "user", "person"}:
            return seller == wanted or buyer == wanted
        if focus in {"property", "land", "blockchainland"}:
            return land_id == wanted
        if focus in {"transaction", "datasettransaction", "blockchaintransaction"}:
            return (
                tx_id == wanted
                or tx_id.endswith(wanted)
                or f"tx {tx_id[-8:]}" == wanted
            )
        return wanted in {tx_id, land_id, seller, buyer}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_graph(
        self,
        limit: int = 600,
        postcode: Optional[str] = None,
        property_id: Optional[str] = None,
        source_filter: str = "all",
        label_filter: str = "all",
        blockchain_transactions: Optional[List[Dict[str, Any]]] = None,
        blockchain_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
        recent_only: bool = False,
        focus_type: Optional[str] = None,
        focus_label: Optional[str] = None,
        admin_only: bool = False,
    ) -> Dict[str, Any]:
        limit = max(1, min(int(limit), 1500))
        postcode_norm = str(postcode or "").strip().casefold()
        property_norm = str(property_id or "").strip().casefold()
        source_filter = str(source_filter or "all").strip().casefold()
        label_filter = str(label_filter or "all").strip().casefold()

        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        dataset = self._load_dataset()

        def add_edge(source: str, target: str, relation: str) -> None:
            edges.append({"source": source, "target": target, "relation": relation})

        # --------------------------------------------------------------
        # Dataset layer
        # --------------------------------------------------------------
        dataset_rows = []
        # admin_only is intentionally a separate network/admin view.
        # The normal user graph always comes from the mixed dataset.
        if not admin_only:
            if focus_label:
                dataset_rows = [
                    row for row in dataset
                    if self._matches_focus(row, focus_type or "", focus_label)
                ]
            elif recent_only:
                dataset_rows = sorted(
                    dataset,
                    key=lambda row: self._timestamp_sort_value(row.get("timestamp")),
                    reverse=True,
                )[:limit]
            else:
                dataset_rows = dataset[:limit]

        selected_rows = 0
        for row in dataset_rows:
            if selected_rows >= limit:
                break

            land_id = str(row.get("land_id") or "").strip()
            if not land_id:
                continue

            if postcode_norm:
                haystack = " ".join(
                    str(row.get(k) or "") for k in ("postcode", "land_id", "address")
                ).casefold()
                if postcode_norm not in haystack:
                    continue

            if property_norm and property_norm not in {
                land_id.casefold(),
                self._property_id(land_id).casefold(),
            }:
                continue

            source = "real_augmented_transactions.json (HM Land Registry + synthetic owners)"
            if source_filter != "all" and source_filter != "dataset":
                continue

            fraud = self._safe_int(row.get("label"), self._safe_int(row.get("is_fraud"), 0))
            if label_filter == "fraud" and fraud != 1:
                continue
            if label_filter == "legitimate" and fraud != 0:
                continue

            tx_id = self._transaction_id(row)
            property_node = self._property_id(land_id)
            tx_node = f"DatasetTx::{tx_id}"

            if property_node not in nodes:
                nodes[property_node] = self._node(
                    property_node,
                    "Property",
                    f"Land {land_id}",
                    land_id=land_id,
                    source=source,
                    dataset=True,
                    owner_source="synthetic_augmented_from_ppd",
                )

            date_label = self._date_label(row.get("timestamp"))
            year = self._year_label(row.get("timestamp"))
            nodes[tx_node] = self._node(
                tx_node,
                "Transaction",
                f"Tx {tx_id[-8:]}",
                transaction_id=tx_id,
                land_id=land_id,
                old_owner=str(row.get("old_owner") or ""),
                new_owner=str(row.get("new_owner") or ""),
                seller=str(row.get("old_owner") or ""),
                buyer=str(row.get("new_owner") or ""),
                price=self._safe_price(row.get("land_price")),
                timestamp=row.get("timestamp"),
                date=date_label,
                year=year,
                document_hash=str(row.get("document_hash") or ""),
                is_fraud=fraud,
                fraud_type=str(row.get("fraud_pattern") or row.get("fraud_type") or ""),
                source=source,
                dataset=True,
            )
            add_edge(property_node, tx_node, "LAND_TRANSACTION")

            for role, owner in (
                ("seller", row.get("old_owner")),
                ("buyer", row.get("new_owner")),
            ):
                owner_name = str(owner or "").strip()
                if not owner_name:
                    continue
                owner_node = f"Owner::{owner_name}"
                if owner_node not in nodes:
                    nodes[owner_node] = self._node(
                        owner_node,
                        "Owner",
                        owner_name,
                        source=source,
                        dataset=True,
                        owner_source="synthetic_augmented_from_ppd",
                    )
                add_edge(owner_node, tx_node, role.upper())

            selected_rows += 1

        # --------------------------------------------------------------
        # Live blockchain/pending layer
        # --------------------------------------------------------------
        blockchain_rows = [] if admin_only else list(blockchain_transactions or [])
        if focus_label and str(focus_type or "").casefold() in {
            "owner", "user", "person", "property", "land", "transaction",
            "datasettransaction", "blockchaintransaction",
        }:
            if str(focus_type or "").casefold() in {"transaction", "datasettransaction", "blockchaintransaction"}:
                wanted = str(focus_label).strip().casefold()
                blockchain_rows = [
                    tx for tx in blockchain_rows
                    if (
                        str(tx.get("id") or "").strip().casefold() == wanted
                        or str(tx.get("id") or "").strip().casefold().endswith(wanted)
                        or f"tx {str(tx.get('id') or '')[-8:]}".casefold() == wanted
                    )
                ]
            else:
                blockchain_rows = [
                    tx for tx in blockchain_rows
                    if self._matches_focus(tx, str(focus_type or ""), focus_label)
                ]

        if recent_only:
            blockchain_rows = sorted(
                blockchain_rows,
                key=lambda tx: self._timestamp_sort_value(tx.get("timestamp")),
                reverse=True,
            )[:limit]

        for tx in blockchain_rows[:limit]:
            land_id = str(tx.get("land_id") or "").strip()
            if not land_id:
                continue

            raw_id = str(tx.get("id") or self._transaction_id(tx)).strip()
            land_node = f"BlockchainLand::{land_id}"
            tx_node = f"BlockchainTx::{raw_id}"

            if land_node not in nodes:
                nodes[land_node] = self._node(
                    land_node,
                    "BlockchainLand",
                    f"Blockchain Land {land_id}",
                    land_id=land_id,
                    source="Blockchain_Ledger.json",
                    blockchain=True,
                )

            date_label = self._date_label(tx.get("timestamp"))
            nodes[tx_node] = self._node(
                tx_node,
                "BlockchainTransaction",
                f"Chain TX {raw_id[-8:]}",
                transaction_id=raw_id,
                land_id=land_id,
                old_owner=str(tx.get("old_owner") or ""),
                new_owner=str(tx.get("new_owner") or ""),
                seller=str(tx.get("old_owner") or ""),
                buyer=str(tx.get("new_owner") or ""),
                price=self._safe_price(tx.get("land_price")),
                timestamp=tx.get("timestamp"),
                date=date_label,
                year=self._year_label(tx.get("timestamp")),
                predicted_consensus=tx.get("predicted_consensus"),
                document_hash=tx.get("document_hash", ""),
                risk_assessment=tx.get("risk_assessment"),
                source=tx.get("source", "ledger"),
                blockchain=True,
            )
            add_edge(land_node, tx_node, "BLOCKCHAIN_RECORD")

            for role, owner in (
                ("seller", tx.get("old_owner")),
                ("buyer", tx.get("new_owner")),
            ):
                owner_name = str(owner or "").strip()
                if not owner_name:
                    continue
                owner_node = f"Owner::{owner_name}"
                if owner_node not in nodes:
                    nodes[owner_node] = self._node(
                        owner_node,
                        "Owner",
                        owner_name,
                        source="Blockchain_Ledger.json",
                        blockchain=True,
                    )
                else:
                    nodes[owner_node]["blockchain"] = True
                add_edge(owner_node, tx_node, role.upper())

            block_index = tx.get("block_index")
            if block_index is not None:
                block_node = f"Block::{block_index}"
                if block_node not in nodes:
                    nodes[block_node] = self._node(
                        block_node,
                        "Block",
                        f"Block #{block_index}",
                        block_index=block_index,
                        source="Blockchain_Ledger.json",
                        blockchain=True,
                    )
                add_edge(block_node, tx_node, "CONTAINS")

        # --------------------------------------------------------------
        # Network nodes
        # --------------------------------------------------------------
        if not focus_label or admin_only:
            for name, meta in (blockchain_nodes or {}).items():
                node_id = f"NetworkNode::{name}"
                nodes[node_id] = self._node(
                    node_id,
                    "NetworkNode",
                    str(name),
                    source="Blockchain_Ledger.json",
                    blockchain=True,
                    stake=meta.get("stake", 0),
                    chain_length=meta.get("chain_length", 0),
                )

        # --------------------------------------------------------------
        # Custom nodes/edges
        # --------------------------------------------------------------
        custom_nodes = list(self.custom_graph.get("nodes", []))
        if recent_only:
            custom_nodes = custom_nodes[-10:]

        # In a focused User/Land/Transaction view, keep only custom nodes that
        # match the search label or link to nodes already in that focused graph.
        if focus_label and not admin_only:
            wanted_label = str(focus_label).strip().casefold()
            current_ids = set(nodes.keys())
            focused_custom = []
            for custom_node in custom_nodes:
                custom_label = str(custom_node.get("label") or "").strip().casefold()
                linked_ids = set(custom_node.get("linked_node_ids", []) or [])
                if custom_label == wanted_label or linked_ids.intersection(current_ids):
                    focused_custom.append(custom_node)
            custom_nodes = focused_custom

        custom_ids = {n.get("id") for n in custom_nodes}
        for node in custom_nodes:
            if node.get("id"):
                nodes[node["id"]] = node
                for linked_id in node.get("linked_node_ids", []) or []:
                    if linked_id in nodes:
                        add_edge(node["id"], linked_id, "USER_LINK")

        for edge in self.custom_graph.get("edges", []):
            if edge.get("source") in custom_ids and edge.get("target") in custom_ids:
                add_edge(
                    edge.get("source"),
                    edge.get("target"),
                    edge.get("relation", "RELATED_TO"),
                )

        # --------------------------------------------------------------
        # Clean duplicate/dangling edges
        # --------------------------------------------------------------
        seen = set()
        clean_edges = []
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            relation = edge.get("relation", "RELATED_TO")
            key = (source, target, relation)
            if not source or not target or key in seen:
                continue
            if source not in nodes or target not in nodes:
                continue
            seen.add(key)
            clean_edges.append({
                "source": source,
                "target": target,
                "relation": relation,
            })

        counts = Counter(node.get("type", "Custom") for node in nodes.values())
        fraud_count = sum(
            1
            for node in nodes.values()
            if node.get("type") == "Transaction" and node.get("is_fraud") == 1
        )

        return {
            "nodes": list(nodes.values()),
            "edges": clean_edges,
            "counts": dict(counts),
            "dataset": {
                "name": "UK HM Land Registry Price Paid Data + Synthetic Owner Details",
                "path": self.dataset_path,
                "available": os.path.exists(self.dataset_path),
                "rows_in_dataset": len(dataset),
                "rows_used_in_graph": selected_rows,
                "recent_only": recent_only,
                "admin_only": admin_only,
                "fraud_records_in_graph": fraud_count,
                "synthetic_transaction_generation_by_graph": False,
                "note": "The graph reads the supplied mixed dataset built from UK HM Land Registry Price Paid Data plus synthetic owner/buyer/seller details. It does not create new transactions or synthetic records.",
                "real_source": "HM Land Registry Price Paid Data",
                "owner_data": "Synthetic owner/buyer/seller augmentation",
            },
        }

    # ------------------------------------------------------------------
    # User-added graph nodes
    # ------------------------------------------------------------------
    def add_custom_node(
        self,
        node_type: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        node_type = str(node_type or "Custom").strip() or "Custom"
        label = str(label or "Untitled").strip() or "Untitled"
        metadata = dict(metadata or {})
        node_id = f"Custom::{uuid.uuid4().hex[:12]}"

        linked_node_ids: List[str] = []
        label_norm = label.casefold()
        focus_type = node_type.casefold()

        for row in self._load_dataset():
            tx_id = self._transaction_id(row)
            tx_node = f"DatasetTx::{tx_id}"
            matches = False
            if focus_type in {"owner", "user", "person"}:
                matches = (
                    str(row.get("old_owner") or "").casefold() == label_norm
                    or str(row.get("new_owner") or "").casefold() == label_norm
                )
            elif focus_type in {"property", "land", "blockchainland"}:
                matches = str(row.get("land_id") or "").casefold() == label_norm
            elif focus_type in {"transaction", "datasettransaction"}:
                raw_id = str(row.get("transaction_id") or row.get("id") or "").casefold()
                matches = raw_id == label_norm or raw_id.endswith(label_norm)

            if matches:
                linked_node_ids.append(tx_node)

        metadata["linked_node_ids"] = linked_node_ids
        metadata["related_transaction_count"] = len(linked_node_ids)

        node = self._node(
            node_id,
            node_type,
            label,
            source="user-added",
            **metadata,
        )
        self.custom_graph.setdefault("nodes", []).append(node)
        self._save_custom_graph()
        return node

    def add_custom_edge(self, source: str, target: str, relation: str) -> Optional[Dict[str, Any]]:
        node_ids = {node.get("id") for node in self.custom_graph.get("nodes", [])}
        if source not in node_ids or target not in node_ids:
            return None
        edge = {
            "source": source,
            "target": target,
            "relation": str(relation or "RELATED_TO").strip() or "RELATED_TO",
        }
        self.custom_graph.setdefault("edges", []).append(edge)
        self._save_custom_graph()
        return edge