"""
=============================================================================
  FAKE PROFILE DETECTION ON SOCIAL MEDIA
  Dataset generation → Preprocessing → Feature Engineering →
  Model Training → Evaluation → Save Artifacts
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import joblib

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

os.makedirs("visualizations", exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ═══════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATASET GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_dataset(n_samples=2000):
    """
    Generate synthetic social media profiles.
    Fake profiles: high following, low followers, no pic, new account.
    Real profiles: balanced ratio, has pic, old account, active.
    """
    print("=" * 60)
    print("  STEP 1: GENERATING SYNTHETIC DATASET")
    print("=" * 60)

    n_real = int(n_samples * 0.55)
    n_fake = n_samples - n_real

    # Real profiles
    real_data = {
        'followers_count': np.random.randint(50, 10000, n_real),
        'following_count': np.random.randint(30, 2000, n_real),
        'posts_count': np.random.randint(10, 1500, n_real),
        'engagement_rate': np.round(np.random.uniform(0.5, 15.0, n_real), 2),
        'profile_picture': np.where(np.random.random(n_real) < 0.92, 1, 0).astype(int),
        'bio_length': np.random.randint(10, 300, n_real),
        'account_age_days': np.random.randint(180, 3650, n_real),
    }
    real_data['fake'] = 0

    # Fake profiles
    fake_data = {
        'followers_count': np.random.randint(0, 200, n_fake),
        'following_count': np.random.randint(500, 7500, n_fake),
        'posts_count': np.random.randint(0, 30, n_fake),
        'engagement_rate': np.round(np.random.uniform(0.0, 2.0, n_fake), 2),
        'profile_picture': np.where(np.random.random(n_fake) < 0.15, 1, 0).astype(int),
        'bio_length': np.random.randint(0, 40, n_fake),
        'account_age_days': np.random.randint(1, 180, n_fake),
    }
    fake_data['fake'] = 1

    df = pd.concat([pd.DataFrame(real_data), pd.DataFrame(fake_data)], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  ✅ Total: {len(df)} | Real: {n_real} | Fake: {n_fake}\n")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. PREPROCESSING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def preprocess_and_engineer_features(df):
    print("=" * 60)
    print("  STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    df.fillna(0, inplace=True)

    df['follower_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
    df['posts_per_day'] = df['posts_count'] / (df['account_age_days'] + 1)
    df['bio_completeness'] = df['bio_length'] * df['profile_picture']
    df['account_freshness'] = 1 / (df['account_age_days'] + 1)
    df['interaction_score'] = df['engagement_rate'] * df['posts_count']

    feature_cols = [
        'followers_count', 'following_count', 'posts_count',
        'engagement_rate', 'profile_picture', 'bio_length',
        'account_age_days', 'follower_following_ratio',
        'posts_per_day', 'bio_completeness',
        'account_freshness', 'interaction_score'
    ]

    X = df[feature_cols]
    y = df['fake']
    print(f"  ✅ Features engineered: {len(feature_cols)} total\n")
    return X, y, feature_cols


# ═══════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

def plot_all(X, y, feature_cols, rf_model, rf_cm, lr_cm, results_df):
    print("=" * 60)
    print("  STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 60 + "\n")

    df_viz = X.copy()
    df_viz['fake'] = y

    # 1. Correlation heatmap
    corr = X[feature_cols].corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualizations/correlation_heatmap.png", dpi=150)
    plt.close()
    print("  ✅ correlation_heatmap.png")

    # 2. Feature distributions
    keys = ['followers_count', 'following_count', 'posts_count',
            'engagement_rate', 'bio_length', 'account_age_days']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, feat in enumerate(keys):
        sns.kdeplot(data=df_viz, x=feat, hue='fake', fill=True,
                    alpha=0.4, ax=axes[i], palette={0: '#2ecc71', 1: '#e74c3c'})
        axes[i].set_title(feat, fontweight='bold')
        axes[i].legend(title="", labels=["Fake", "Real"])
    fig.suptitle("Feature Distributions: Fake vs Real", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("visualizations/distribution_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ distribution_plots.png")

    # 3. Engagement vs fake scatter
    df_viz['ff_ratio'] = df_viz['followers_count'] / (df_viz['following_count'] + 1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='ff_ratio', y='engagement_rate', hue='fake',
                    palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.5, s=40)
    plt.title("Engagement Rate vs Follower/Following Ratio", fontsize=13, fontweight='bold')
    plt.xlabel("Follower/Following Ratio")
    plt.ylabel("Engagement Rate (%)")
    plt.legend(title="", labels=["Fake", "Real"])
    plt.tight_layout()
    plt.savefig("visualizations/engagement_vs_fake.png", dpi=150)
    plt.close()
    print("  ✅ engagement_vs_fake.png")

    # 4 & 5. Confusion matrices
    for cm, name, fname in [(rf_cm, "Random Forest", "confusion_matrix_rf.png"),
                             (lr_cm, "Logistic Regression", "confusion_matrix_lr.png")]:
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                    linewidths=1, linecolor='gray', annot_kws={"size": 14, "fontweight": "bold"})
        plt.title(f"Confusion Matrix — {name}", fontsize=13, fontweight='bold')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"visualizations/{fname}", dpi=150)
        plt.close()
        print(f"  ✅ {fname}")

    # 6. Model comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.30
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, results_df.iloc[0][metrics], width,
                label='Random Forest', color='#2ecc71', edgecolor='black')
    b2 = ax.bar(x + width/2, results_df.iloc[1][metrics], width,
                label='Logistic Regression', color='#3498db', edgecolor='black')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/model_comparison.png", dpi=150)
    plt.close()
    print("  ✅ model_comparison.png")

    # 7. Feature importance
    imp = pd.DataFrame({'Feature': feature_cols,
                        'Importance': rf_model.feature_importances_})
    imp = imp.sort_values('Importance', ascending=True)
    plt.figure(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(imp)))
    plt.barh(imp['Feature'], imp['Importance'], color=colors, edgecolor='black', linewidth=0.5)
    plt.title("Random Forest — Feature Importance", fontsize=13, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("visualizations/feature_importance.png", dpi=150)
    plt.close()
    print("  ✅ feature_importance.png\n")


# ═══════════════════════════════════════════════════════════════
# 4. TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate(X, y, feature_cols):
    print("=" * 60)
    print("  STEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  → Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    print("  🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                min_samples_split=5, min_samples_leaf=2,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_preds = rf.predict(X_test_s)

    # Logistic Regression
    print("  📈 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    lr_preds = lr.predict(X_test_s)

    # Metrics
    print("\n" + "=" * 60)
    print("  STEP 4: EVALUATION RESULTS")
    print("=" * 60 + "\n")

    rf_cm = confusion_matrix(y_test, rf_preds)
    lr_cm = confusion_matrix(y_test, lr_preds)

    results = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression'],
        'Accuracy': [accuracy_score(y_test, rf_preds), accuracy_score(y_test, lr_preds)],
        'Precision': [precision_score(y_test, rf_preds), precision_score(y_test, lr_preds)],
        'Recall': [recall_score(y_test, rf_preds), recall_score(y_test, lr_preds)],
        'F1-Score': [f1_score(y_test, rf_preds), f1_score(y_test, lr_preds)]
    })

    print(f"  {'Metric':<20} {'Random Forest':>15} {'Logistic Reg':>15}")
    print(f"  {'─' * 52}")
    for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        print(f"  {m:<20} {results.iloc[0][m]:>15.4f} {results.iloc[1][m]:>15.4f}")

    print("\n  ── Random Forest Report ──")
    print(classification_report(y_test, rf_preds, target_names=['Real', 'Fake']))
    print("  ── Logistic Regression Report ──")
    print(classification_report(y_test, lr_preds, target_names=['Real', 'Fake']))

    # Visualize
    plot_all(X, y, feature_cols, rf, rf_cm, lr_cm, results)

    # Save
    joblib.dump(rf, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("  💾 model.pkl + scaler.pkl saved\n")

    return results, rf


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   FAKE PROFILE DETECTION — TRAINING PIPELINE    ║")
    print("╚══════════════════════════════════════════════════╝\n")

    df = generate_synthetic_dataset(2000)
    df.to_csv("fake_profiles.csv", index=False)
    print("  💾 fake_profiles.csv saved\n")

    X, y, feature_cols = preprocess_and_engineer_features(df)
    results, model = train_and_evaluate(X, y, feature_cols)

    print("=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"""
  📊 Dataset      : {len(df)} profiles
  🧪 Features     : {len(feature_cols)}
  🌲 Best Model   : Random Forest
  🎯 Accuracy     : {results.iloc[0]['Accuracy']:.4f}
  📈 F1-Score     : {results.iloc[0]['F1-Score']:.4f}
  📁 Graphs       : 7 files in /visualizations/
  🚀 Next         : python server.py
    """)