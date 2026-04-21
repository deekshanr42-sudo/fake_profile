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

    # Return features in the exact same order as training
    features = np.array([[
        followers, following, posts, engagement,
        has_pic, bio_len, account_age,
        ff_ratio, posts_per_day, bio_completeness,
        freshness, interaction
    ]])

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

    viz_files = {
        "Feature Correlation Heatmap": "correlation_heatmap.png",
        "Feature Distributions": "distribution_plots.png",
        "Engagement vs Fake Analysis": "engagement_vs_fake.png",
        "Confusion Matrix (RF)": "confusion_matrix_rf.png",
        "Confusion Matrix (LR)": "confusion_matrix_lr.png",
        "Model Comparison": "model_comparison.png",
        "Feature Importance": "feature_importance.png",
    }

    selected_viz = st.selectbox("Choose graph:", list(viz_files.keys()))

    viz_path = os.path.join("visualizations", viz_files[selected_viz])
    if os.path.exists(viz_path):
        img = Image.open(viz_path)
        st.image(img, caption=selected_viz, use_column_width=True)
    else:
        st.info("Run `train_model.py` to generate visualizations.")

    st.markdown("---")
    st.markdown("🧑‍💻 Built with **Streamlit** & **Scikit-Learn**")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🔍 Fake Profile Detector</h1>
    <p>Detect fake social media profiles using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    💡 <strong>How it works:</strong> Enter the profile details below and our
    Random Forest model will analyze behavioral patterns to predict if the
    profile is <strong>Fake</strong> or <strong>Real</strong>.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════

if MODEL_LOADED:

    st.markdown("### 📝 Enter Profile Details")

    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        followers = st.number_input(
            "👥 Followers Count", min_value=0, max_value=1000000,
            value=150, help="Total number of followers"
        )
        following = st.number_input(
            "👤 Following Count", min_value=0, max_value=1000000,
            value=300, help="Total number of accounts followed"
        )
        posts = st.number_input(
            "📝 Posts Count", min_value=0, max_value=100000,
            value=50, help="Total number of posts published"
        )
        engagement = st.number_input(
            "💬 Engagement Rate (%)", min_value=0.0, max_value=100.0,
            value=3.5, step=0.1, help="Average likes+comments per post / followers × 100"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        has_pic = st.selectbox(
            "📸 Profile Picture", options=["Yes", "No"],
            help="Does the profile have a profile picture?"
        )
        bio_len = st.number_input(
            "📖 Bio Length (characters)", min_value=0, max_value=5000,
            value=120, help="Number of characters in the profile bio"
        )
        account_age = st.number_input(
            "📅 Account Age (days)", min_value=0, max_value=36500,
            value=365, help="How many days old is this account?"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Convert profile picture to binary
    has_pic_binary = 1 if has_pic == "Yes" else 0

    # ── PREDICT BUTTON ──
    st.markdown("---")
    predict_btn = st.button("🚀 Analyze Profile", type="primary",
                            use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # PREDICTION & RESULT DISPLAY
    # ═══════════════════════════════════════════════════════════════════

    if predict_btn:

        # Engineer features (same as training)
        features = engineer_features(
            followers, following, posts, engagement,
            has_pic_binary, bio_len, account_age
        )

        # Scale features
        features_scaled = scaler.transform(features)

        # Get prediction and probability
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        # Confidence is the max probability
        confidence = max(probability) * 100

        # Display result
        if prediction == 1:
            # FAKE
            st.markdown(f"""
            <div class="result-card result-fake">
                <div class="result-emoji">🚨</div>
                <div class="result-text">FAKE PROFILE</div>
                <div class="result-confidence">
                    Confidence: {confidence:.1f}% | Probability of being fake: {probability[1]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show warning signs
            st.markdown("#### ⚠️ Potential Red Flags Detected:")
            warnings_list = []
            if followers / (following + 1) < 0.3:
                warnings_list.append("🔴 Very low follower-to-following ratio")
            if posts < 10:
                warnings_list.append("🔴 Very few posts published")
            if has_pic_binary == 0:
                warnings_list.append("🔴 No profile picture")
            if bio_len < 20:
                warnings_list.append("🔴 Very short or empty bio")
            if account_age < 90:
                warnings_list.append("🔴 Account is less than 3 months old")
            if engagement < 1.0:
                warnings_list.append("🔴 Extremely low engagement rate")

            if warnings_list:
                for w in warnings_list:
                    st.markdown(w)
            else:
                st.markdown("🟡 Multiple subtle patterns indicate suspicious activity.")
        else:
            # REAL
            st.markdown(f"""
            <div class="result-card result-real">
                <div class="result-emoji">✅</div>
                <div class="result-text">REAL PROFILE</div>
                <div class="result-confidence">
                    Confidence: {confidence:.1f}% | Probability of being real: {probability[0]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### ✅ Profile appears legitimate based on:")
            green_flags = []
            if followers / (following + 1) > 0.3:
                green_flags.append("🟢 Healthy follower-to-following ratio")
            if posts > 10:
                green_flags.append("🟢 Consistent posting history")
            if has_pic_binary == 1:
                green_flags.append("🟢 Profile picture present")
            if bio_len > 20:
                green_flags.append("🟢 Detailed bio information")
            if account_age > 180:
                green_flags.append("🟢 Established account age")
            if engagement > 1.0:
                green_flags.append("🟢 Healthy engagement rate")

            for g in green_flags:
                st.markdown(g)

        # ── Show feature values used ──
        with st.expander("🔧 View Engineered Features Used"):
            ff_ratio = followers / (following + 1)
            ppd = posts / (account_age + 1)
            bio_comp = bio_len * has_pic_binary
            fresh = 1 / (account_age + 1)
            inter = engagement * posts

            feat_df = pd.DataFrame({
                'Feature': [
                    'Followers', 'Following', 'Posts', 'Engagement Rate',
                    'Has Profile Pic', 'Bio Length', 'Account Age (days)',
                    'Follower/Following Ratio', 'Posts Per Day',
                    'Bio Completeness', 'Account Freshness', 'Interaction Score'
                ],
                'Value': [
                    followers, following, posts, engagement,
                    has_pic_binary, bio_len, account_age,
                    round(ff_ratio, 4), round(ppd, 4),
                    bio_comp, round(fresh, 6), round(inter, 2)
                ]
            })
            st.dataframe(feat_df, hide_index=True, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # SAMPLE PROFILES SECTION
    # ═══════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("### 🧪 Try Sample Profiles")

    sample_col1, sample_col2 = st.columns(2)

    with sample_col1:
        if st.button("👤 Load Real Profile Example", use_container_width=True):
            st.session_state['sample'] = 'real'
            st.rerun()

    with sample_col2:
        if st.button("🤖 Load Fake Profile Example", use_container_width=True):
            st.session_state['sample'] = 'fake'
            st.rerun()

    if 'sample' in st.session_state:
        if st.session_state['sample'] == 'real':
            st.info("""
            **Real Profile Example:**  
            - 5,000 followers | 800 following | 500 posts  
            - Engagement: 6.5% | Has picture | Bio: 180 chars  
            - Account age: 730 days (2 years)
            """)
        else:
            st.info("""
            **Fake Profile Example:**  
            - 25 followers | 3,500 following | 3 posts  
            - Engagement: 0.2% | No picture | Bio: 5 chars  
            - Account age: 15 days
            """)

else:
    st.error("""
    ## ⚠️ Model Not Found
    Please run the training script first:
    ```
    python train_model.py
    ```
    Then refresh this page.
    """)