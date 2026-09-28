import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

st.set_page_config(page_title="Student Wellbeing Predictor", page_icon="🧠", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);
       padding: 2rem 2.2rem; border-radius: 20px; color: white; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: .4rem 0 0; opacity: .9; font-size: 1.05rem;}
.card {background: rgba(124,58,237,.07); border: 1px solid rgba(124,58,237,.25);
       border-radius: 16px; padding: 1.1rem 1.3rem; margin-bottom: .8rem;}
.card h4 {margin: 0 0 .3rem; font-size: .9rem; opacity: .7; font-weight: 500;}
.card .val {font-size: 1.8rem; font-weight: 700;}
.tip {border-left: 4px solid #7c3aed; padding: .6rem 1rem; margin: .5rem 0;
      background: rgba(124,58,237,.06); border-radius: 0 10px 10px 0;}
div.stButton > button {background: linear-gradient(90deg,#4f46e5,#db2777); color: white;
      border: 0; border-radius: 12px; padding: .7rem 1rem; font-weight: 600; width: 100%;}
div.stButton > button:hover {filter: brightness(1.1); color: white;}
</style>
""", unsafe_allow_html=True)

# ---------- Data ----------
DATA_CANDIDATES = ["student social media health impact.zip",
                   "student social media health impact.csv",
                   "data.csv"]

@st.cache_data(show_spinner=False)
def load_data(path_or_file):
    return pd.read_csv(path_or_file)

def get_data():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            return load_data(p)
    st.info("Dataset not found next to app.py. Upload it to continue.")
    up = st.file_uploader("Upload the student social media dataset (.csv or .zip)",
                          type=["csv", "zip"])
    if up is None:
        st.stop()
    return load_data(up)

# ---------- Model (same pipeline as the notebook) ----------
SKEWED = ["Study_Hours"]
NUMERIC = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks",
           "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
ORDINAL = ["Stress_Level"]
NOMINAL = ["Gender", "Academic_Level", "Most_Used_Platform",
           "Purpose_Of_Use", "Grouped_country"]
STRESS_ORDER = ["Low", "Medium", "High", "Very High"]

@st.cache_resource(show_spinner="Training the model…")
def train(df: pd.DataFrame):
    df = df.drop_duplicates().copy()
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
    top = df["Country"].value_counts().index[:10].tolist()
    df["Grouped_country"] = df["Country"].apply(lambda c: c if c in top else "Other")

    X = df[SKEWED + NUMERIC + ORDINAL + NOMINAL]
    y = df["Mental_Health_Score"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    pre = ColumnTransformer([
        ("skew", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("sc", StandardScaler())]), SKEWED),
        ("num", StandardScaler(), NUMERIC),
        ("ord", OrdinalEncoder(categories=[STRESS_ORDER]), ORDINAL),
        ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
    ])
    model = Pipeline([("pre", pre), ("rf", RandomForestRegressor(
        random_state=42, n_estimators=300, max_depth=15,
        min_samples_split=5, min_samples_leaf=2, n_jobs=-1))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"r2": r2_score(y_te, pred), "mae": mean_absolute_error(y_te, pred),
               "train_r2": r2_score(y_tr, model.predict(X_tr))}

    names = model.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names)
    imp.index = [n.split("__", 1)[1] for n in imp.index]
    return model, metrics, imp, top, df, y_te, pred

df_raw = get_data()
model, metrics, importances, top_countries, df, y_te, y_pred = train(df_raw)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🧠 Student Wellbeing Predictor</h1>
  <p>Estimate a student's mental health score from social media habits, sleep, stress and lifestyle.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("👤 Student profile")
    age = st.slider("Age", 18, 24, 21)
    gender = st.radio("Gender", sorted(df["Gender"].unique()), horizontal=True)
    level = st.selectbox("Academic level", sorted(df["Academic_Level"].unique()))
    country = st.selectbox("Country", sorted(df["Country"].unique()))

    st.header("📱 Social media")
    platform = st.selectbox("Most used platform", sorted(df["Most_Used_Platform"].unique()))
    purpose = st.selectbox("Main purpose", sorted(df["Purpose_Of_Use"].unique()))
    usage = st.slider("Daily usage (hours)", 1.0, 9.0, 5.0, 0.1)
    unlocks = st.slider("Daily phone unlocks", 50, 280, 170)

    st.header("🌿 Lifestyle")
    study = st.slider("Study hours / day", 0.3, 8.5, 3.0, 0.1)
    activity = st.slider("Physical activity (hours)", 0.0, 4.0, 1.5, 0.1)
    sleep = st.slider("Sleep (hours / night)", 3.0, 10.0, 7.0, 0.1)
    stress = st.select_slider("Stress level", STRESS_ORDER, value="Medium")

    go_btn = st.button("✨ Predict score")

sample = pd.DataFrame([{
    "Study_Hours": study, "Age": age, "Avg_Daily_Usage_Hours": usage,
    "Daily_Unlocks": unlocks, "Physical_Activity_Hours": activity,
    "Sleep_Hours_Per_Night": sleep, "Stress_Level": stress, "Gender": gender,
    "Academic_Level": level, "Most_Used_Platform": platform,
    "Purpose_Of_Use": purpose,
    "Grouped_country": country if country in top_countries else "Other",
}])

tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Data insights", "🤖 Model performance"])

# ---------- Tab 1: Prediction ----------
with tab1:
    score = float(np.clip(model.predict(sample)[0], 0, 10))
    avg = df["Mental_Health_Score"].mean()

    left, right = st.columns([1.2, 1])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=score,
            number={"font": {"size": 54}, "valueformat": ".1f"},
            delta={"reference": avg, "valueformat": ".1f", "suffix": " vs avg"},
            title={"text": "Predicted mental health score (higher is better)"},
            gauge={"axis": {"range": [0, 10]}, "bar": {"color": "#7c3aed"},
                   "steps": [{"range": [0, 5], "color": "rgba(239,68,68,.35)"},
                             {"range": [5, 7], "color": "rgba(245,158,11,.35)"},
                             {"range": [7, 10], "color": "rgba(34,197,94,.35)"}],
                   "threshold": {"line": {"color": "#db2777", "width": 4}, "value": avg}}))
        fig.update_layout(height=340, margin=dict(t=70, b=10, l=30, r=30))
        st.plotly_chart(fig)
    with right:
        band = "Good 🟢" if score >= 7 else "Moderate 🟡" if score >= 5 else "At risk 🔴"
        st.markdown(f'<div class="card"><h4>Wellbeing band</h4><div class="val">{band}</div></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="card"><h4>Dataset average</h4><div class="val">{avg:.2f}</div></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="card"><h4>Typical model error</h4><div class="val">± {metrics["mae"]:.2f}</div></div>',
                    unsafe_allow_html=True)

    st.subheader("💡 What stands out")
    tips = []
    if sleep < 6.5: tips.append(f"Sleep is **{sleep:.1f} h**, below the ~{df['Sleep_Hours_Per_Night'].mean():.1f} h average. More sleep tends to go with higher scores.")
    if usage > df["Avg_Daily_Usage_Hours"].quantile(.75): tips.append(f"Daily usage of **{usage:.1f} h** is in the top 25% of students.")
    if stress in ("High", "Very High"): tips.append(f"**{stress}** stress is strongly linked with lower scores in this data.")
    if activity < 1: tips.append("Very little physical activity. Even a short daily routine could help.")
    if not tips: tips.append("Nothing unusual in this profile. Habits look balanced.")
    for t in tips:
        st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)
    st.caption("This is a statistical estimate from a survey dataset, not a medical assessment.")

# ---------- Tab 2: Insights ----------
with tab2:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(df):,}")
    c2.metric("Avg daily usage", f"{df['Avg_Daily_Usage_Hours'].mean():.1f} h")
    c3.metric("Avg sleep", f"{df['Sleep_Hours_Per_Night'].mean():.1f} h")
    c4.metric("Avg mental score", f"{df['Mental_Health_Score'].mean():.2f}")

    a, b = st.columns(2)
    with a:
        f = px.histogram(df, x="Mental_Health_Score", nbins=20, marginal="box",
                         title="Mental health score distribution",
                         color_discrete_sequence=["#7c3aed"])
        st.plotly_chart(f)
    with b:
        g = df.groupby("Most_Used_Platform", as_index=False)["Avg_Daily_Usage_Hours"].mean() \
              .sort_values("Avg_Daily_Usage_Hours")
        f = px.bar(g, x="Avg_Daily_Usage_Hours", y="Most_Used_Platform", orientation="h",
                   title="Average daily usage by platform", color="Avg_Daily_Usage_Hours",
                   color_continuous_scale="Purples")
        f.update_layout(coloraxis_showscale=False)
        st.plotly_chart(f)

    a, b = st.columns(2)
    with a:
        f = px.box(df, x="Stress_Level", y="Mental_Health_Score",
                   category_orders={"Stress_Level": STRESS_ORDER}, color="Stress_Level",
                   title="Mental health score by stress level")
        f.update_layout(showlegend=False)
        st.plotly_chart(f)
    with b:
        f = px.scatter(df.sample(min(1500, len(df)), random_state=1),
                       x="Sleep_Hours_Per_Night", y="Mental_Health_Score",
                       color="Avg_Daily_Usage_Hours", color_continuous_scale="Plasma",
                       opacity=.7, title="Sleep vs mental health (colored by usage)")
        st.plotly_chart(f)

    corr = df.corr(numeric_only=True)
    f = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                  zmin=-1, zmax=1, title="Correlation heatmap", aspect="auto")
    st.plotly_chart(f)

# ---------- Tab 3: Model ----------
with tab3:
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (test)", f"{metrics['r2']:.3f}")
    m2.metric("R² (train)", f"{metrics['train_r2']:.3f}")
    m3.metric("MAE (test)", f"{metrics['mae']:.3f}")

    a, b = st.columns(2)
    with a:
        top = importances.sort_values().tail(12)
        f = px.bar(x=top.values, y=top.index, orientation="h",
                   title="Top feature importances", color=top.values,
                   color_continuous_scale="Purples", labels={"x": "Importance", "y": ""})
        f.update_layout(coloraxis_showscale=False)
        st.plotly_chart(f)
    with b:
        f = px.scatter(x=y_te, y=y_pred, opacity=.5, title="Actual vs predicted",
                       labels={"x": "Actual", "y": "Predicted"},
                       color_discrete_sequence=["#7c3aed"])
        lo, hi = float(y_te.min()), float(y_te.max())
        f.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#db2777", dash="dash"))
        st.plotly_chart(f)

    st.caption("Random Forest (300 trees, depth 15) inside a preprocessing pipeline: log + scaling for "
               "Study_Hours, scaling for numeric columns, ordinal encoding for stress, one-hot for categories.")