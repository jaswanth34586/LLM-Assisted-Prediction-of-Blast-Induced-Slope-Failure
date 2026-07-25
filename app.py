#!/usr/bin/env python3
"""
BISF Prediction System — Web Server (Pure Python, no external deps)
Run:  python3 app.py
"""

import http.server
import json
import math
import sys
import webbrowser
import urllib.parse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

from config.config import *
from src.utils import create_sample_dataset, calculate_ppv
from src.data_preprocessing import DataPreprocessor
from src.baseline_models import BaselineModels
from src.evaluation import ModelEvaluator
from src.llm_workflow import (
    get_feature_suggestions, extract_rules_from_tree,
    explain_prediction, get_workflow_summary, format_feature_prompt
)

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
PORT = 8081

# ── Global state ───────────────────────────────────────────────────────────
STATE = {
    "models": {},
    "results": {},
    "feature_names": [],
    "X_test": None,
    "y_test": None,
    "X_train": None,
    "y_train": None,
    "preprocessor": None,
    "data_path": None,
    "trained": False,
    "llm_rules": None,
}

MIME_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


# ── Chart data helpers ─────────────────────────────────────────────────────
def get_roc_data():
    roc = {}
    for name, model in STATE["models"].items():
        proba = model.predict_proba(STATE["X_test"])
        y_scores = [p[1] for p in proba]
        y_true = STATE["y_test"]

        # compute ROC curve
        pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        fpr_list, tpr_list = [0.0], [0.0]
        tp, fp = 0, 0
        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            fpr_list.append(fp / n_neg)
            tpr_list.append(tp / n_pos)

        # AUC via trapezoidal
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

        # downsample
        step = max(1, len(fpr_list) // 100)
        roc[name] = {
            "fpr": [round(fpr_list[i], 4) for i in range(0, len(fpr_list), step)],
            "tpr": [round(tpr_list[i], 4) for i in range(0, len(tpr_list), step)],
            "auc": round(auc, 4),
        }
    return roc


def get_feature_data():
    best = max(STATE["results"], key=lambda x: STATE["results"][x]['fbeta_score'])
    model = STATE["models"][best]
    imp = model.feature_importances_
    if not imp:
        return {"model": best, "names": [], "values": []}

    indexed = sorted(enumerate(imp), key=lambda x: x[1])[-10:]
    return {
        "model": best,
        "names": [STATE["feature_names"][i] for i, _ in indexed],
        "values": [round(v, 5) for _, v in indexed],
    }


# ── API handlers ───────────────────────────────────────────────────────────
def api_upload_data(handler):
    content_type = handler.headers.get('Content-Type', '')
    if 'multipart/form-data' not in content_type:
        return {"ok": False, "message": "Invalid upload"}

    content_length = int(handler.headers.get('Content-Length', 0))
    body = handler.rfile.read(content_length)

    boundary = None
    for part in content_type.split(';'):
        part = part.strip()
        if part.startswith('boundary='):
            boundary = part.split('=', 1)[1].strip('"')
            break
    if not boundary:
        return {"ok": False, "message": "No boundary in multipart data"}

    delimiter = f'--{boundary}'.encode()
    parts = body.split(delimiter)
    file_data = None
    for part in parts:
        if b'name="csvfile"' in part:
            header_end = part.find(b'\r\n\r\n')
            if header_end != -1:
                file_data = part[header_end + 4:]
                if file_data.endswith(b'\r\n'):
                    file_data = file_data[:-2]
            break

    if not file_data:
        return {"ok": False, "message": "No file received"}

    path = RAW_DATA_DIR / "blast_data.csv"
    path.write_bytes(file_data)
    STATE["data_path"] = path
    print(f"✓ Uploaded dataset → {path}")
    return {"ok": True, "message": f"Dataset uploaded ({path.name})"}


def api_train(body):
    if not STATE["data_path"]:
        return {"ok": False, "message": "Upload a dataset first"}

    data = json.loads(body)
    selected = data.get("models", [])
    if not selected:
        return {"ok": False, "message": "Select at least one model"}

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.full_pipeline(STATE["data_path"])

    STATE["X_train"] = X_train
    STATE["X_test"] = X_test
    STATE["y_train"] = y_train
    STATE["y_test"] = y_test
    STATE["llm_rules"] = None
    STATE["feature_names"] = feature_names
    STATE["preprocessor"] = preprocessor

    baseline = BaselineModels()
    baseline.initialize_models()
    evaluator = ModelEvaluator()

    STATE["models"] = {}
    STATE["results"] = {}

    for name in selected:
        if name in baseline.models:
            model = baseline.train_model(name, X_train, y_train)
            STATE["models"][name] = model
            y_pred = model.predict(X_test)
            proba = model.predict_proba(X_test)
            y_proba = [p[1] for p in proba]
            STATE["results"][name] = evaluator.evaluate(y_test, y_pred, y_proba, name)

    STATE["trained"] = True
    print(f"✓ Trained {len(selected)} models")

    results_json = {}
    for name, r in STATE["results"].items():
        results_json[name] = {k: v for k, v in r.items() if isinstance(v, (int, float))}

    best_model = max(STATE["results"], key=lambda x: STATE["results"][x]['fbeta_score'])
    return {
        "ok": True,
        "message": f"Trained {len(selected)} models successfully",
        "results": results_json,
        "best_model": best_model,
    }


def api_dashboard():
    if not STATE["trained"]:
        return {"ok": False, "message": "Train models first"}

    best_name = max(STATE["results"], key=lambda x: STATE["results"][x]['fbeta_score'])
    best = STATE["results"][best_name]

    metric_keys = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'fbeta_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'F-beta']
    chart_data = {}
    for name, r in STATE["results"].items():
        chart_data[name] = [round(r[m], 4) for m in metric_keys]

    # Confusion matrix data as JSON (rendered by Chart.js in frontend)
    cm_data = {}
    for name, r in STATE["results"].items():
        cm_data[name] = {"tn": r['tn'], "fp": r['fp'], "fn": r['fn'], "tp": r['tp']}

    return {
        "ok": True,
        "metrics": [
            {"label": f"Accuracy ({best_name})", "value": f"{best['accuracy']:.3f}"},
            {"label": f"Recall ({best_name})", "value": f"{best['recall']:.3f}"},
            {"label": f"Specificity ({best_name})", "value": f"{best['specificity']:.3f}"},
            {"label": f"F-beta ({best_name})", "value": f"{best['fbeta_score']:.3f}"},
        ],
        "chart_labels": metric_labels,
        "chart_data": chart_data,
        "cm_data": cm_data,
    }


def api_comparison():
    if not STATE["trained"]:
        return {"ok": False, "message": "Train models first"}
    return {
        "ok": True,
        "roc_data": get_roc_data(),
        "feat_data": get_feature_data(),
    }


def api_llm_insights():
    if not STATE["trained"]:
        return {"ok": False, "message": "Train models first"}

    suggestions = get_feature_suggestions()

    if STATE["llm_rules"] is None:
        STATE["llm_rules"] = extract_rules_from_tree(
            STATE["X_train"], STATE["y_train"], STATE["feature_names"]
        )

    best_name = max(STATE["results"], key=lambda x: STATE["results"][x]['fbeta_score'])
    best_model = STATE["models"][best_name]
    sample = STATE["X_test"][0]
    pred = best_model.predict([sample])[0]
    proba = best_model.predict_proba([sample])[0]
    conf = proba[1] if pred == 1 else proba[0]
    row_dict = dict(zip(STATE["feature_names"], sample))
    explanation = explain_prediction(
        row_dict, "FAILURE" if pred == 1 else "SAFE", conf
    )

    workflow = get_workflow_summary()

    return {
        "ok": True,
        "suggestions": suggestions,
        "rules": STATE["llm_rules"]["rules"],
        "tree_text": STATE["llm_rules"]["tree_text"],
        "tree_accuracy": STATE["llm_rules"]["tree_accuracy"],
        "top_features": STATE["llm_rules"]["top_features"],
        "explanation": explanation,
        "workflow": workflow,
        "prompt": format_feature_prompt(),
    }


# ── HTTP Handler ───────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/dashboard":
            return self._json_response(api_dashboard())
        if path == "/api/comparison":
            return self._json_response(api_comparison())
        if path == "/api/llm_insights":
            return self._json_response(api_llm_insights())

        if path == "/":
            path = "/static/index.html"

        if path.startswith("/static/"):
            file_path = STATIC_DIR / path[len("/static/"):]
            if file_path.is_file():
                ext = file_path.suffix
                mime = MIME_TYPES.get(ext, "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.end_headers()
                self.wfile.write(file_path.read_bytes())
                return

        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/upload_data":
            result = api_upload_data(self)
            return self._json_response(result)

        if path == "/api/train":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode()
            result = api_train(body)
            return self._json_response(result)

        self._json_response({"ok": False, "message": "Unknown endpoint"})

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = http.server.HTTPServer(("", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n{'=' * 50}")
    print(f"  BISF Prediction System")
    print(f"  Running at: {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'=' * 50}\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()