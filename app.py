"""
=============================================================================
BNA FraudShield — Flask API Server  (v3.1 — persistent DB + auth)
app.py
=============================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import sqlite3, os, hmac, hashlib, json

from fraud_model import predict, THRESHOLD_FRAUD, THRESHOLD_REVIEW

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# ─────────────────────────────────────────────
# AUTH CONFIG  (admin / bna)
# ─────────────────────────────────────────────
ADMIN_USER = "admin"
ADMIN_PASS = "bna"
SECRET_KEY = "bna_fraudshield_2025_pfe_secret"

def _make_token(username, password):
    raw = f"{username}:{password}:{SECRET_KEY}"
    return hmac.new(SECRET_KEY.encode(), raw.encode(), hashlib.sha256).hexdigest()

VALID_TOKEN = _make_token(ADMIN_USER, ADMIN_PASS)

def _is_admin(req):
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return hmac.compare_digest(token, VALID_TOKEN)
    return False

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraudshield.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS validated_transactions (
            tx_id      TEXT PRIMARY KEY,
            tx_data    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fraud_analyses (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_id      TEXT NOT NULL,
            tx_data    TEXT NOT NULL,
            result     TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
# CORS preflight
# ─────────────────────────────────────────────
@app.route("/api/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return jsonify({}), 200

# ─────────────────────────────────────────────
# ENDPOINT: POST /api/auth/login
# ─────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    if username == ADMIN_USER and password == ADMIN_PASS:
        return jsonify({"success": True, "token": VALID_TOKEN, "role": "admin"})
    return jsonify({"success": False, "error": "Invalid credentials"}), 401

# ─────────────────────────────────────────────
# ENDPOINT: POST /api/transactions/save  (public — called after validation)
# ─────────────────────────────────────────────
@app.route("/api/transactions/save", methods=["POST"])
def api_save_transaction():
    data = request.get_json(force=True) or {}
    tx_id = data.get("txId", "")
    if not tx_id:
        return jsonify({"success": False, "error": "Missing txId"}), 400
    try:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO validated_transactions (tx_id, tx_data) VALUES (?, ?)",
            (tx_id, json.dumps(data))
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ─────────────────────────────────────────────
# ENDPOINT: GET /api/transactions/pool  (admin)
# ─────────────────────────────────────────────
@app.route("/api/transactions/pool", methods=["GET"])
def api_get_pool():
    if not _is_admin(request):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT tx_data FROM validated_transactions ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        transactions = [json.loads(r["tx_data"]) for r in rows]
        return jsonify({"success": True, "transactions": transactions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ─────────────────────────────────────────────
# ENDPOINT: POST /api/analyses/save  (admin)
# ─────────────────────────────────────────────
@app.route("/api/analyses/save", methods=["POST"])
def api_save_analysis():
    if not _is_admin(request):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    data = request.get_json(force=True) or {}
    tx_id  = data.get("txId", "")
    tx_data = data.get("txData", {})
    result  = data.get("result", {})
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO fraud_analyses (tx_id, tx_data, result) VALUES (?, ?, ?)",
            (tx_id, json.dumps(tx_data), json.dumps(result))
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ─────────────────────────────────────────────
# ENDPOINT: GET /api/analyses/history  (admin)
# ─────────────────────────────────────────────
@app.route("/api/analyses/history", methods=["GET"])
def api_get_history():
    if not _is_admin(request):
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT tx_id, tx_data, result, created_at FROM fraud_analyses ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        analyses = []
        for r in rows:
            item = json.loads(r["tx_data"])
            item.update(json.loads(r["result"]))
            item["txId"]      = r["tx_id"]
            item["savedAt"]   = r["created_at"]
            analyses.append(item)
        return jsonify({"success": True, "analyses": analyses})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ─────────────────────────────────────────────
# EXISTING ENDPOINTS (unchanged)
# ─────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "Empty request body"}), 400
        required = ["amount", "balance", "hour", "tx_type", "device", "location"]
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({"success": False, "error": f"Missing: {', '.join(missing)}"}), 400
        result = predict(data)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status"          : "online",
        "model"           : "XGBoost Ensemble v2 — BNA calibrated",
        "dataset"         : "200,000 BNA transactions (Star Schema)",
        "fraud_rate"      : "5.04%",
        "imbalance_ratio" : "18.8:1",
        "auc_roc"         : 0.96,
        "threshold_fraud" : THRESHOLD_FRAUD,
        "threshold_review": THRESHOLD_REVIEW,
        "version"         : "3.1.0",
        "server_time"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.route("/api/weekend", methods=["GET"])
def api_weekend():
    date_str = request.args.get("date", "")
    if not date_str:
        return jsonify({"error": "Missing 'date' query parameter"}), 400
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        wd = dt.weekday()
        french_days = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        return jsonify({
            "date"        : date_str,
            "day_of_week" : french_days[wd],
            "is_weekend"  : 1 if wd >= 5 else 0,
            "month"       : dt.month,
            "day_of_month": dt.day,
        })
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

@app.route("/", methods=["GET"])
def index():
    return "<h3>BNA FraudShield API v3.1</h3><p>POST /api/predict | GET /api/health | POST /api/auth/login</p>"

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

if __name__ == "__main__":
    print("=" * 55)
    print("  BNA FraudShield v3.1 — ML API + Auth + DB")
    print("  Running on http://localhost:5000")
    print("=" * 55)
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
