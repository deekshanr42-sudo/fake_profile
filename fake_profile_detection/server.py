"""
=============================================================================
  FAKE PROFILE DETECTION — FLASK BACKEND API
  ──────────────────────────────────────────
  Serves the trained ML model as a REST API.
  Endpoints:
    GET  /api/health        → Health check
    POST /api/predict       → Predict fake/real from profile data
    GET  /api/visualizations → List available visualization files
    GET  /viz/<filename>    → Serve a visualization image
=============================================================================
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os

# ── Initialize Flask App ──
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Allow cross-origin requests from frontend

# ── Load Model and Scaler ──
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
VIZ_DIR = "visualizations"

model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please run 'python train_model.py' first.")


def engineer_features(followers, following, posts, engagement,
                      has_pic, bio_len, account_age):
    """
    Create the same 5 derived features used during training.
    Returns a pandas DataFrame with the expected feature columns.
    """
    ff_ratio = followers / (following + 1)
    posts_per_day = posts / (account_age + 1)
    bio_completeness = bio_len * has_pic
    freshness = 1 / (account_age + 1)
    interaction = engagement * posts

    return pd.DataFrame({
        'followers_count': [followers],
        'following_count': [following],
        'posts_count': [posts],
        'engagement_rate': [engagement],
        'profile_picture': [has_pic],
        'bio_length': [bio_len],
        'account_age_days': [account_age],
        'follower_following_ratio': [ff_ratio],
        'posts_per_day': [posts_per_day],
        'bio_completeness': [bio_completeness],
        'account_freshness': [freshness],
        'interaction_score': [interaction]
    })


# ═══════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def serve_index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/health')
def health_check():
    """Check if the API and model are ready."""
    if model is None or scaler is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "ok", "message": "Model is ready"})


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accept profile data as JSON, return prediction.
    
    Expected JSON body:
    {
        "followers": 150,
        "following": 300,
        "posts": 50,
        "engagement_rate": 3.5,
        "has_profile_picture": true,
        "bio_length": 120,
        "account_age_days": 365
    }
    
    Returns:
    {
        "prediction": "Real" | "Fake",
        "confidence": 97.5,
        "probability_fake": 0.025,
        "probability_real": 0.975,
        "red_flags": [...],
        "green_flags": [...]
    }
    """
    if model is None or scaler is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Run train_model.py first."
        }), 503

    try:
        data = request.get_json()

        # Extract and validate fields
        followers = float(data.get('followers', 0))
        following = float(data.get('following', 0))
        posts = float(data.get('posts', 0))
        engagement = float(data.get('engagement_rate', 0))
        has_pic = 1 if data.get('has_profile_picture', False) else 0
        bio_len = float(data.get('bio_length', 0))
        account_age = float(data.get('account_age_days', 1))

        # Engineer features
        features = engineer_features(
            followers, following, posts, engagement,
            has_pic, bio_len, account_age
        )

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        prob_real = float(probabilities[0])
        prob_fake = float(probabilities[1])
        confidence = max(prob_real, prob_fake) * 100
        label = "Fake" if prediction == 1 else "Real"

        # Generate flags
        red_flags = []
        green_flags = []
        ff_ratio = followers / (following + 1)

        if ff_ratio < 0.3:
            red_flags.append("Very low follower-to-following ratio")
        else:
            green_flags.append("Healthy follower-to-following ratio")

        if posts < 10:
            red_flags.append("Very few posts published")
        else:
            green_flags.append("Consistent posting history")

        if has_pic == 0:
            red_flags.append("No profile picture")
        else:
            green_flags.append("Profile picture present")

        if bio_len < 20:
            red_flags.append("Very short or empty bio")
        else:
            green_flags.append("Detailed bio information")

        if account_age < 90:
            red_flags.append("Account is less than 3 months old")
        else:
            green_flags.append("Established account age")

        if engagement < 1.0:
            red_flags.append("Extremely low engagement rate")
        else:
            green_flags.append("Healthy engagement rate")

        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": round(confidence, 1),
            "probability_fake": round(prob_fake, 4),
            "probability_real": round(prob_real, 4),
            "red_flags": red_flags,
            "green_flags": green_flags,
            "features_used": {
                "follower_following_ratio": round(ff_ratio, 4),
                "posts_per_day": round(posts / (account_age + 1), 4),
                "bio_completeness": int(bio_len * has_pic),
                "account_freshness": round(1 / (account_age + 1), 6),
                "interaction_score": round(engagement * posts, 2)
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route('/api/visualizations')
def list_visualizations():
    """Return list of available visualization files."""
    if not os.path.exists(VIZ_DIR):
        return jsonify({"files": []})
    files = [f for f in os.listdir(VIZ_DIR) if f.endswith('.png')]
    return jsonify({"files": files})


@app.route('/viz/<filename>')
def serve_visualization(filename):
    """Serve a visualization image file."""
    return send_from_directory(VIZ_DIR, filename)


# ═══════════════════════════════════════════════════════════════════
#  START SERVER
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  🔍 Fake Profile Detector — Flask Server")
    print("  🌐 Open: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000)