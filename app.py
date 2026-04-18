"""
BNA FraudShield — Flask API Server (Render-ready version)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os

# Import the ML scoring engine
from fraud_model import predict, THRESHOLD_FRAUD, THRESHOLD_REVIEW

app = Flask(__name__)
CORS(app)   # Allow requests from index.html (Netlify)

# =============================================================================
# MAIN ENDPOINT: POST /api/predict
# =============================================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"success": False, "error": "Empty request body"}), 400

        result = predict(data)
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =============================================================================
# HEALTH CHECK
# =============================================================================
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "online",
        "model": "XGBoost Ensemble v2 — BNA calibrated",
        "dataset": "200,000 BNA transactions",
        "fraud_rate": "5.04%",
        "version": "2.1.0",
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

# =============================================================================
# WEEKEND HELPER (for frontend)
# =============================================================================
@app.route("/api/weekend", methods=["GET"])
def api_weekend():
    date_str = request.args.get("date", "")
    if not date_str:
        return jsonify({"error": "Missing 'date' query parameter"}), 400
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        wd = dt.weekday()
        french_days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        return jsonify({
            "date": date_str,
            "day_of_week": french_days[wd],
            "is_weekend": 1 if wd >= 5 else 0,
            "month": dt.month,
            "day_of_month": dt.day,
        })
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

# =============================================================================
# START SERVER — Render uses $PORT environment variable
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 BNA FraudShield API running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)