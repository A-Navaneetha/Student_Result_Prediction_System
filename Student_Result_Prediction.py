import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from datetime import datetime

# PAGE CONFIG
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="wide"
)

# CSS
st.markdown(
    """
    <style>
        [data-testid="stHeader"] {display: none;}
        div[class^="block-container"] {padding-top: 0rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #0ea5e9
    );
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
    margin: auto;
}
.main-title {
    font-size: 3rem; font-weight: 900; text-align: center;
    color: #e0f2fe; text-shadow: 0 6px 30px rgba(14,165,233,0.35);
}
.sub-title { text-align: center; color: #bae6fd; font-size: 1.1rem; margin-bottom: 25px; }
.metric-card {
    background: rgba(255,255,255,0.15); backdrop-filter: blur(12px);
    border-radius: 15px; padding: 20px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2); transition: 0.3s;
}            
.stats-card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    display: flex;
    align-items: center;
    gap: 15px;
    height: 120px;
}
.stats-icon {
    width: 70px;
    height: 70px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
}
.stats-title {
    font-size: 18px;
    font-weight: 600;
    color: #334155;
}
.stats-value {
    font-size: 38px;
    font-weight: 800;
    margin-top: 5px;
}
.metric-card:hover { transform: translateY(-5px); }
.pass-badge {
    background: linear-gradient(135deg, #22c55e, #16a34a); color: white;
    border-radius: 15px; padding: 25px; font-size: 2rem; font-weight: bold;
    text-align: center; box-shadow: 0px 10px 25px rgba(34,197,94,0.4);
}
.fail-badge {
    background: linear-gradient(135deg, #ef4444, #dc2626); color: white;
    border-radius: 15px; padding: 25px; font-size: 2rem; font-weight: bold;
    text-align: center; box-shadow: 0px 10px 25px rgba(239,68,68,0.4);
}
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #06b6d4); color: white ;
    border: none; border-radius: 12px; height: 3.2em;
    font-size: 18px; font-weight: bold; transition: 0.3s;
}
.stButton > button:hover { transform: scale(1.05); box-shadow: 0px 5px 15px rgba(0,0,0,0.3); }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.15); border-radius: 10px; margin-right: 5px; color: white;
}
.stTabs [aria-selected="true"] { background: #2563eb; color: white; }
.info-box {
    background: rgba(255,255,255,0.12); backdrop-filter: blur(10px);
    border-left: 5px solid #38bdf8; padding: 15px; border-radius: 10px; color: white;
}
[data-testid="stDataFrame"] { background: rgba(255,255,255,0.1); border-radius: 15px; overflow: hidden; }
.footer { text-align: center; color: #cbd5e1; margin-top: 20px; font-size: 14px; }

/* ── Headings & text inside tabs ── */
h1, h2, h3, h4, h5, h6 { color: #e0f2fe !important; }
.stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4 { color: #e0f2fe !important; }
[data-testid="stSubheader"],
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 { color: #e0f2fe !important; }

/* st.subheader() */
div[data-testid="stHeadingWithActionElements"] > div > div,
div[data-testid="stHeadingWithActionElements"] p { color: #e0f2fe !important; }

/* All plain paragraph text inside tabs */
.stMarkdown p, .stMarkdown li,
div[data-testid="stMarkdownContainer"] p { color: #e0f2fe !important; }

/* st.write() and st.info() labels */
div[data-testid="stText"] { color: #e0f2fe !important; }
div[data-testid="stInfo"]  { color: #0c4a6e !important; }

/* Slider labels */
label[data-testid="stWidgetLabel"] p { color: #bae6fd !important; }

/* Metric labels and values */
div[data-testid="stMetricLabel"]  p { color: #93c5fd !important; }
div[data-testid="stMetricValue"]  > div { color: #e0f2fe !important; }

/* st.write() confidence text */
div[data-testid="stMarkdownContainer"] { color: #e0f2fe; }

/* Expander header text */
details summary span { color: #bae6fd !important; }
details summary p    { color: #bae6fd !important; }

/* metric-card value paragraph */
.metric-card h3 { color: #93c5fd !important; font-size: 0.95rem; margin: 0 0 6px; }
.metric-card p  { color: #ffffff !important; font-size: 1.8rem; font-weight: 700; margin: 0; }
</style>
""", unsafe_allow_html=True)

# MATPLOTLIB GLOBAL DARK THEME
plt.rcParams.update({
    "figure.facecolor":  "none",
    "axes.facecolor":    "#1e3a8a22",
    "axes.edgecolor":    "#93c5fd",
    "axes.titlecolor":   "#e0f2fe",
    "axes.labelcolor":   "#bae6fd",
    "xtick.color":       "#bae6fd",
    "ytick.color":       "#bae6fd",
    "text.color":        "#e0f2fe",
    "legend.facecolor":  "#0f172a",
    "legend.edgecolor":  "#334155",
    "legend.labelcolor": "#e0f2fe",
    "grid.color":        "#334155",
})

# DATA GENERATION
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500
    hours_study    = np.random.uniform(1, 12, n)
    attendance     = np.random.uniform(40, 100, n)
    previous_marks = np.random.uniform(30, 100, n)
    assignments    = np.random.randint(0, 11, n)
    sleep_hours    = np.random.uniform(4, 10, n)
    marks = (
        0.30 * previous_marks +
        0.25 * (hours_study / 12 * 100) +
        0.20 * attendance +
        0.15 * (assignments / 10 * 100) +
        0.10 * (sleep_hours / 10 * 100) +
        np.random.normal(0, 5, n)
    )
    marks = np.clip(marks, 0, 100)
    result = (marks >= 50).astype(int)
    df = pd.DataFrame({
        "Hours_Study":      hours_study,
        "Attendance":       attendance,
        "Previous_Marks":   previous_marks,
        "Assignments_Done": assignments,
        "Sleep_Hours":      sleep_hours,
        "Marks":            marks,
        "Result":           result
    })
    return df

# MODEL TRAINING
@st.cache_resource
def train_model(df):
    features = ["Hours_Study", "Attendance", "Previous_Marks",
                "Assignments_Done", "Sleep_Hours"]
    X = df[features].values
    y = df["Result"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "cm":        confusion_matrix(y_test, y_pred)
    }
    return model, scaler, metrics

# PDF REPORT GENERATOR
def create_pdf_report(student_data, prediction, confidence, recommendations):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Student Result Prediction Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(
    Paragraph(
        f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')} ",
        styles['Italic']
    )
    )
    elements.append(
    Paragraph(
        " By Student Result Prediction System",
        styles['Italic']
    )
    )
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Student Details</b>", styles['Heading2']))
    for key, value in student_data.items():
        elements.append(Paragraph(f"{key}: {value}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles['Heading2']))
    elements.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "Developed By: A.Navaneetha",
        styles['Italic']
    ))
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# LOAD DATA & MODEL
df = generate_data()
model, scaler, metrics = train_model(df)
features = ["Hours_Study", "Attendance", "Previous_Marks", "Assignments_Done", "Sleep_Hours"]

# HEADER
st.markdown('<div class="main-title">🎓 Student Result Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">ML-powered PASS / FAIL predictor using Logistic Regression'
    ' &nbsp;|&nbsp; Python & Streamlit</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# TABS
tab1, tab2, tab3 = st.tabs(["🔮 Predict Result", "📊 Data Analysis", "📈 Model Performance"])

# ══════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════
with tab1:
    st.subheader("Enter Student Details")
    st.write("Adjust the sliders and click **Predict** to get the result.")

    col1, col2 = st.columns(2)
    with col1:
        hours       = st.slider("📚 Study Hours per Day",   1.0, 12.0,  6.0, 0.5)
        attendance  = st.slider("🏫 Attendance (%)",        40.0, 100.0, 75.0, 1.0)
        prev_marks  = st.slider("📝 Previous Exam Marks",   30.0, 100.0, 60.0, 1.0)
    with col2:
        assignments = st.slider("📋 Assignments Completed", 0, 10, 5)
        sleep       = st.slider("😴 Average Sleep Hours",   4.0, 10.0,  7.0, 0.5)

    st.markdown("")
    predict_btn = st.button("🔍 Predict Result", use_container_width=True)

    if predict_btn:
        inp  = scaler.transform([[hours, attendance, prev_marks, assignments, sleep]])
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        pass_probability = prob[1] * 100

        # Performance category
        if pass_probability >= 90:
            category = "⭐ Excellent"
        elif pass_probability >= 75:
            category = "👍 Good"
        elif pass_probability >= 60:
            category = "📚 Average"
        elif pass_probability >= 50:
            category = "⚠️ Needs Improvement"
        else:
            category = "🚨 At Risk"

        # Recommendations
        recommendations = []
        if attendance < 70:
            recommendations.append("Improve attendance above 70%.")
        if hours < 5:
            recommendations.append("Increase study hours by 2–3 hours daily.")
        if assignments < 5:
            recommendations.append("Submit more assignments regularly.")
        if sleep < 6:
            recommendations.append("Maintain at least 7 hours of sleep.")
        if prev_marks < 60:
            recommendations.append("Focus on strengthening core subjects.")
        if not recommendations:
            recommendations.append("Excellent performance. Keep up the good work!")

        st.markdown("---")
        col_r, col_s = st.columns(2)

        with col_r:
            if pred == 1:
                st.markdown(
                    f'<div class="pass-badge">✅ PASS &nbsp;({prob[1]*100:.1f}% confidence)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="fail-badge">❌ FAIL &nbsp;({prob[0]*100:.1f}% confidence)</div>',
                    unsafe_allow_html=True
                )
            st.info(f"Performance Level: {category}")

        with col_s:
            summary = pd.DataFrame({
                "Parameter": ["Study Hours", "Attendance (%)", "Previous Marks",
                              "Assignments Done", "Sleep Hours"],
                "Value":     [hours, attendance, prev_marks, assignments, sleep]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        # Probability bar
        st.markdown("#### Probability Breakdown")
        fig_p, ax_p = plt.subplots(figsize=(6, 1.2))
        fig_p.patch.set_alpha(0.0)
        ax_p.set_facecolor("#1e3a8a22")
        for spine in ax_p.spines.values():
            spine.set_edgecolor("#93c5fd")
        ax_p.barh(["Probability"], [prob[0]], color="#e74c3c", label="FAIL")
        ax_p.barh(["Probability"], [prob[1]], left=[prob[0]], color="#2ecc71", label="PASS")
        ax_p.set_xlim(0, 1)
        ax_p.set_xlabel("Probability")
        ax_p.legend(loc="lower right")
        ax_p.set_title(f"FAIL: {prob[0]*100:.1f}%   |   PASS: {prob[1]*100:.1f}%")
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close()

        # Confidence bar
        st.markdown("### 📈 Confidence Score")
        st.progress(pass_probability / 100)
        st.write(f"Prediction Confidence: {pass_probability:.2f}%")

        st.markdown("""
        <div class="info-box">
        💡 <b>Tip:</b> Students who study more than 7 hours, maintain 80%+ attendance,
        and complete all assignments have the highest pass rates in our dataset.
        </div>""", unsafe_allow_html=True)
    
        st.markdown("### Recommendations")
        for rec in recommendations:
            st.success(rec)

        student_data = {
            "Study Hours":    hours,
            "Attendance":     attendance,
            "Previous Marks": prev_marks,
            "Assignments":    assignments,
            "Sleep Hours":    sleep
        }

        pdf = create_pdf_report(
            student_data,
            "PASS" if pred == 1 else "FAIL",
            max(prob) * 100,
            recommendations
        )

        st.download_button(
            label="📄 Download Prediction Report",
            data=pdf,
            file_name="Student_Result_Report.pdf",
            mime="application/pdf"
        )

# ══════════════════════════════════════
# TAB 2 — DATA ANALYSIS
# ══════════════════════════════════════
with tab2:

    st.subheader("Dataset Insights")
    total = len(df)
    passed = int(df["Result"].sum())
    failed = total - passed
    rate = passed / total * 100
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-icon" style="background:#dbeafe;">👥</div>
            <div>
                <div class="stats-title">Total Students</div>
                <div class="stats-value" style="color:#2563eb;">{total}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-icon" style="background:#dcfce7;">✅</div>
            <div>
                <div class="stats-title">Pass Count</div>
                <div class="stats-value" style="color:#16a34a;">{passed}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-icon" style="background:#fee2e2;">❌</div>
            <div>
                <div class="stats-title">Fail Count</div>
                <div class="stats-value" style="color:#dc2626;">{failed}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-icon" style="background:#f3e8ff;">📊</div>
            <div>
                <div class="stats-title">Pass Rate</div>
                <div class="stats-value" style="color:#9333ea;">{rate:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    scatter_colors = df["Result"].map({0: "red", 1: "green"})

# Chart 1

    axes[0,0].scatter(
    df["Hours_Study"],
    df["Marks"],
    c=scatter_colors,
    alpha=0.6
    )
    axes[0,0].set_title("Study Hours vs Marks")
    axes[0,0].set_xlabel("Study Hours")
    axes[0,0].set_ylabel("Marks")

# Chart 2
    axes[0,1].scatter(
    df["Attendance"],
    df["Marks"],
    c=scatter_colors,
    alpha=0.6
    )
    axes[0,1].set_title("Attendance vs Marks")
    axes[0,1].set_xlabel("Attendance")
    axes[0,1].set_ylabel("Marks")

# Chart 3
# ==========================
# PASS / FAIL PIE CHART
# ==========================
    ax = axes[1, 0]
    ax.set_facecolor("#ffffff")
    wedges, texts, autotexts = ax.pie(
    [passed, failed],
    labels=["Pass", "Fail"],
    colors=["#22c55e", "#ef4444"],
    autopct="%1.1f%%",
    startangle=140,
    textprops={
        "color": "black",
        "fontsize": 12,
        "fontweight": "bold"
    },
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    }
    )
    ax.set_title(
    "Pass / Fail Distribution",
    fontsize=14,
    fontweight="bold",
    color="white"
    )

# Chart 4
# ==========================
# MARKS DISTRIBUTION
# ==========================
    ax = axes[1, 1]
    ax.set_facecolor("#ffffff")
    ax.hist(
    df[df["Result"] == 1]["Marks"],
    bins=20,
    color="#22c55e",
    alpha=0.8,
    label="Pass",
    edgecolor="black"
    )
    ax.hist(
    df[df["Result"] == 0]["Marks"],
    bins=20,
    color="#ef4444",
    alpha=0.8,
    label="Fail",
    edgecolor="black"
    )
    ax.axvline(
    50,
    color="#2563eb",
    linestyle="--",
    linewidth=2,
    label="Pass Mark (50)"
    )
    ax.set_xlabel(
    "Marks",
    color="black",
    fontsize=12,
    fontweight="bold"
    )
    ax.set_ylabel(
    "Number of Students",
    color="black",
    fontsize=12,
    fontweight="bold"
    )
    ax.tick_params(colors="black")
    ax.set_title(
    "Marks Distribution",
    fontsize=14,
    fontweight="bold",
    color="white"
    )
    ax.legend(
    facecolor="white",
    edgecolor="black"
    )
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("## 🏆 Top 10 Students")

    top_students = df.sort_values("Marks", ascending=False).head(10)

    st.dataframe(
        top_students[
            ["Hours_Study", "Attendance", "Previous_Marks", "Marks"]
        ].round(2),
        use_container_width=True
    )

    with st.expander("🔍 View Raw Dataset (first 20 rows)"):
        display_df = df.copy()
        display_df["Result_Label"] = display_df["Result"].map(
            {1: "✅ Pass", 0: "❌ Fail"}
        )
        st.dataframe(
            display_df.head(20).round(2),
            use_container_width=True,
            hide_index=True
        )

# ══════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════
with tab3:
    st.subheader("Model Performance Metrics")

    c1, c2, c3, c4 = st.columns(4)
    metric_map = {
        "🎯 Accuracy":  metrics["Accuracy"],
        "🔬 Precision": metrics["Precision"],
        "📡 Recall":    metrics["Recall"],
        "⚖️ F1 Score":  metrics["F1 Score"],
    }
    for col, (label, val) in zip([c1, c2, c3, c4], metric_map.items()):
        col.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <p style="font-size:1.8rem;font-weight:700">{val*100:.2f}%</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_alpha(0.0)
        sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted Fail", "Predicted Pass"],
                    yticklabels=["Actual Fail", "Actual Pass"],
                    linewidths=0.5, ax=ax_cm,
                    annot_kws={"size": 14, "weight": "bold", "color": "black"})
    
        ax_cm.set_title("Confusion Matrix", color="#e0f2fe", fontsize=14, fontweight="bold")
        ax_cm.tick_params(colors="#bae6fd", labelsize=9)
        ax_cm.xaxis.label.set_color("#bae6fd")
        ax_cm.yaxis.label.set_color("#bae6fd")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with col_fi:
        st.markdown("#### Feature Importance (Coefficients)")
        coef    = model.coef_[0]
        feat_df = pd.DataFrame({"Feature": features, "Coefficient": coef})
        feat_df = feat_df.reindex(feat_df["Coefficient"].abs().sort_values(ascending=True).index)
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        bar_colors = ["#2ecc71" if c >= 0 else "#e74c3c" for c in feat_df["Coefficient"]]
        ax_fi.barh(feat_df["Feature"], feat_df["Coefficient"],
                   color=bar_colors, edgecolor="white")
        ax_fi.axvline(0, color="#e0f2fe", linewidth=0.8)
        ax_fi.set_xlabel("Coefficient Value", color="#bae6fd")
        ax_fi.set_title("Feature Importance", color="#e0f2fe", fontweight="bold")
        ax_fi.tick_params(colors="#bae6fd")
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

    st.markdown("---")
    st.markdown("#### 📖 What Do These Metrics Mean?")
    explanations = {
        "🎯 Accuracy":  "The percentage of students the model classified correctly out of all students.",
        "🔬 Precision": "When the model predicts PASS, how often it is actually correct. High precision = fewer false alarms.",
        "📡 Recall":    "Of all students who actually passed, how many did the model successfully identify? High recall = fewer missed passes.",
        "⚖️ F1 Score":  "The harmonic mean of Precision and Recall — a single balanced score that combines both.",
    }
    for label, explanation in explanations.items():
        st.markdown(f"""
        <div class="info-box">
        <b>{label}:</b> {explanation}
        </div>""", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    🛠️ <b>Model Details:</b> Logistic Regression &nbsp;|&nbsp; Solver: lbfgs &nbsp;|&nbsp;
    Max Iterations: 1000 &nbsp;|&nbsp; Train/Test Split: 80 / 20 &nbsp;|&nbsp;
    Feature Scaling: StandardScaler &nbsp;|&nbsp; Dataset: 500 synthetic records
    </div>""", unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Streamlit &amp; Machine Learning Web Application</p>
    <p>Developed by <b>A.Navaneetha</b> |&nbsp; June 2026</p>
</div>
""", unsafe_allow_html=True)