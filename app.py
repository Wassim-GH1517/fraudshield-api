"""
=============================================================================
BNA FraudShield — Flask API Server
app.py
=============================================================================
Web server that exposes the fraud detection model as a REST API.
The ML logic lives entirely in fraud_model.py — this file only handles
HTTP routing, request parsing, and response formatting.

Endpoints:
  POST /api/predict    →  Score a transaction
  GET  /api/health     →  Model status
  GET  /api/weekend    →  Date info (weekend detection for frontend)

Run:
  pip install flask flask-cors
  python app.py

Then open index.html in your browser.
The sidebar will show "API Connected" when the frontend detects this server.
=============================================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Import the ML scoring engine
from fraud_model import predict, THRESHOLD_FRAUD, THRESHOLD_REVIEW


app = Flask(__name__)
CORS(app)   # Allow requests from the HTML frontend (index.html)


# =============================================================================
# ENDPOINT: POST /api/predict
# Accepts a transaction JSON body, returns fraud prediction
# =============================================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Score a transaction for fraud.

    Request body (JSON):
    {
        "amount"       : 2500,
        "balance"      : 1800,
        "hour"         : 2,
        "month"        : 12,
        "is_weekend"   : 1,
        "tx_type"      : "Transfer",
        "account_type" : "Business",
        "device"       : "ATM",
        "location"     : "Tunisia, Agence Testour, Route de Béja, Testour"
    }

    Response (JSON):
    {
        "success"    : true,
        "result"     : {
            "probability"     : 0.7823,
            "probability_pct" : 78.2,
            "verdict"         : "FRAUD",
            "action"          : "BLOCK",
            "contributions"   : [...],
            "log_odds"        : 1.28,
            "features"        : {...},
            "model_info"      : {...}
        }
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "Empty request body"}), 400

        # Required fields validation
        required = ["amount", "balance", "hour", "tx_type", "device", "location"]
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error"  : f"Missing required fields: {', '.join(missing)}"
            }), 400

        # Call the ML scoring engine
        result = predict(data)

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# ENDPOINT: GET /api/health
# Returns model status — used by frontend to detect API connectivity
# =============================================================================
@app.route("/api/health", methods=["GET"])
def api_health():
    """Return model health and metadata."""
    return jsonify({
        "status"          : "online",
        "model"           : "XGBoost Ensemble v2 — BNA calibrated",
        "dataset"         : "200,000 BNA transactions (Star Schema)",
        "fraud_rate"      : "5.04%",
        "imbalance_ratio" : "18.8:1",
        "auc_roc"         : 0.96,
        "threshold_fraud" : THRESHOLD_FRAUD,
        "threshold_review": THRESHOLD_REVIEW,
        "version"         : "2.1.0",
        "server_time"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


# =============================================================================
# ENDPOINT: GET /api/weekend?date=YYYY-MM-DD
# Helper endpoint used by the frontend date picker for auto-detection
# =============================================================================
@app.route("/api/weekend", methods=["GET"])
def api_weekend():
    """
    Return day-of-week and weekend status for a given date.
    Used by the frontend to auto-populate the Is Weekend field.

    Query param: date=YYYY-MM-DD
    """
    date_str = request.args.get("date", "")
    if not date_str:
        return jsonify({"error": "Missing 'date' query parameter"}), 400

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        wd = dt.weekday()   # 0=Monday … 6=Sunday
        french_days = [
            "Lundi", "Mardi", "Mercredi", "Jeudi",
            "Vendredi", "Samedi", "Dimanche"
        ]
        return jsonify({
            "date"        : date_str,
            "day_of_week" : french_days[wd],
            "is_weekend"  : 1 if wd >= 5 else 0,
            "month"       : dt.month,
            "day_of_month": dt.day,
        })

    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400


# =============================================================================
# ENDPOINT: GET /
# Simple status page
# =============================================================================
@app.route("/", methods=["GET"])
def index():
    return (
        "<h3>BNA FraudShield API</h3>"
        "<p>POST /api/predict — Score a transaction<br>"
        "GET  /api/health   — Model status<br>"
        "GET  /api/weekend?date=YYYY-MM-DD — Weekend detection</p>"
    )


# =============================================================================
# START SERVER
# =============================================================================
if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  BNA FraudShield — ML API Server")
    print("  Model: XGBoost Ensemble v2")
    print("  Dataset: 200,000 transactions | AUC-ROC: 0.96")
    print("=" * 55)
    print("  Running on http://localhost:5000")
    print("  Open index.html in your browser")
    print("=" * 55)
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
