"""
=============================================================================
  FAKE PROFILE DETECTION — STREAMLIT WEB APPLICATION
  ──────────────────────────────────────────────────
  This app allows users to input social media profile details
  and get an instant prediction of whether the profile is
  Fake or Real using the trained Random Forest model.
=============================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st                    # Web framework
import pandas as pd                       # Data handling
import numpy as np                        # Math operations
import joblib                             # Load saved model
from PIL import Image                     # Display images
import os                                 # File paths

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fake Profile Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════

custom_css = """
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e74c3c, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #7f8c8d;
    }

    /* Result card styling */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .result-fake {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        box-shadow: 0 8px 25px rgba(238, 90, 36, 0.3);
    }
    .result-real {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        box-shadow: 0 8px 25px rgba(39, 174, 96, 0.3);
    }
    .result-emoji {
        font-size: 4rem;
        margin-bottom: 10px;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: 800;
    }
    .result-confidence {
        font-size: 1.1rem;
        margin-top: 8px;
        opacity: 0.9;
    }

    /* Input section styling */
    .input-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    /* Info box */
    .info-box {
        background: #eaf2f8;
        border-left: 4px solid #3498db;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        font-size: 0.95rem;
        color: #2c3e50;
    }

    /* Sidebar styling */
    .sidebar-section {
        margin-bottom: 20px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL AND SCALER
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Load the trained model and scaler (cached for performance)."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_model()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False
    st.error("⚠️ **Model files not found!** Please run `train_model.py` first.")

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTION (must match training pipeline)
# ═══════════════════════════════════════════════════════════════════════════


def engineer_features(followers, following, posts, engagement,
                      has_pic, bio_len, account_age):
    """
    Create the same derived features used during training.
    This ensures the model receives input in the expected format.
    """

    # Follower-Following Ratio
    ff_ratio = followers / (following + 1)

    # Posts Per Day
    posts_per_day = posts / (account_age + 1)

    # Bio Completeness
    bio_completeness = bio_len * has_pic

    # Account Freshness
    freshness = 1 / (account_age + 1)

    # Interaction Score
    interaction = engagement * posts

    # Return features in the exact same format as training
    features = pd.DataFrame({
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

    return features


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — About & Visualizations
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown("## 📋 About This Project")
    st.markdown("""
    This tool uses **Machine Learning** to detect fake social media
    profiles based on behavioral patterns.

    **Model Used:** Random Forest Classifier  
    **Features Analyzed:** 12 (7 raw + 5 engineered)  
    **Training Accuracy:** ~97%+
    """)

    st.markdown("---")

    st.markdown("## 📊 View Visualizations")
    st.markdown("Select a graph to display:")
