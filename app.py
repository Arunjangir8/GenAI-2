"""
Run with:  streamlit run app.py
"""

import streamlit as st
import time
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="RealAdvisor AI | Agentic Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;700&display=swap');

:root {
    --bg:         #0d0f14;
    --surface:    #161922;
    --surface2:   #1e2230;
    --border:     #2a2f42;
    --accent:     #6c63ff;
    --accent2:    #00c9a7;
    --warn:       #f7b731;
    --danger:     #ee5a24;
    --text:       #e4e6f0;
    --text-muted: #7f8699;
}

html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; font-family: 'Inter', sans-serif; }

.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headers */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-success { border-left: 3px solid var(--accent2); }
.card-warn    { border-left: 3px solid var(--warn); }

/* Metric tiles */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 1rem 0;
}
.metric-tile {
    background: var(--surface2);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid var(--border);
}
.metric-tile .value { font-size: 1.6rem; font-weight: 700; color: var(--accent2); font-family: 'Space Grotesk'; }
.metric-tile .label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }

/* Step tracker */
.step-tracker { display: flex; gap: 8px; margin-bottom: 1.5rem; flex-wrap: wrap; }
.step-badge {
    padding: 6px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    border: 1px solid var(--border); color: var(--text-muted); background: var(--surface);
}
.step-badge.done    { background: #1a3d30; border-color: var(--accent2); color: var(--accent2); }
.step-badge.active  { background: #2d2766; border-color: var(--accent); color: #fff; }

/* Recommendation badge */
.rec-badge {
    display: inline-block; font-size: 1.3rem; font-weight: 800;
    padding: 8px 28px; border-radius: 8px; margin-bottom: 1rem;
    font-family: 'Space Grotesk';
}
.rec-buy    { background: #1a3d30; color: #00e5a0; border: 2px solid #00c9a7; }
.rec-hold   { background: #3d3010; color: #f5c842; border: 2px solid #f7b731; }
.rec-avoid  { background: #3d1010; color: #ff6b5b; border: 2px solid #ee5a24; }

/* Inputs */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSlider"] { background: var(--surface2) !important; border-color: var(--border) !important; color: var(--text) !important; }

/* Button */
.stButton button {
    background: linear-gradient(135deg, var(--accent), #8b5cf6) !important;
    color: white !important; border: none !important;
    padding: 14px 32px !important; border-radius: 10px !important;
    font-size: 1rem !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important; width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.88 !important; }

/* Tab */
button[data-baseweb="tab"] { color: var(--text-muted) !important; background: transparent !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


from config import VALID_CITIES, VALID_PROPERTY_TYPES, VALID_STATUSES, CITY_META
from predictor import format_inr


CITY_COORDS = {
    "Delhi":  {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Pune":   {"lat": 18.5204, "lon": 73.8567},
}

CITY_LOCATIONS = {
    "Delhi":  ["Dwarka", "Rohini", "Saket", "Janakpuri", "Vasant Kunj", "Lajpat Nagar", "South Extension", "Karol Bagh"],
    "Mumbai": ["Andheri", "Powai", "Dadar", "Thane", "Malad", "Navi Mumbai", "Bandra", "Goregaon"],
    "Pune":   ["Hinjewadi", "Kharadi", "Kothrud", "Baner", "Wakad", "Hadapsar", "Viman Nagar", "Undri"],
}

STEP_LABELS = [
    "Input Validation", "Price Prediction", "Market Data (RAG)",
    "Comparable Analysis", "Risk Assessment", "Investment Advice", "Report Compilation"
]


def step_tracker_html(current_step_idx: int) -> str:
    badges = ""
    for i, label in enumerate(STEP_LABELS):
        if i < current_step_idx:
            badges += f'<div class="step-badge done">✓ {label}</div>'
        elif i == current_step_idx:
            badges += f'<div class="step-badge active">⟳ {label}</div>'
        else:
            badges += f'<div class="step-badge">{label}</div>'
    return f'<div class="step-tracker">{badges}</div>'


def metric_tile(value: str, label: str) -> str:
    return f"""
    <div class="metric-tile">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>
    """

def get_rec_class(advice_text: str) -> str:
    text_upper = advice_text.upper()
    if "BUY" in text_upper[:200]:    return "rec-buy",  "BUY"
    if "AVOID" in text_upper[:200]:  return "rec-avoid", "AVOID"
    return "rec-hold", "HOLD"


def make_comparison_chart(comps, predicted_rent):
    """Plotly bar chart of comparable rents."""
    names = [c["name"].split(" ")[-2] + " " + c["name"].split(" ")[-1] for c in comps]
    rents = [c["rent"] for c in comps]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=rents,
        marker_color=["#6c63ff"] * len(names),
        name="Comparable Rent",
        text=[f"₹{r:,}" for r in rents],
        textposition="outside",
    ))
    fig.add_hline(
        y=predicted_rent, line_dash="dot", line_color="#00c9a7", line_width=2,
        annotation_text=f"Predicted ₹{predicted_rent:,.0f}", annotation_font_color="#00c9a7"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e4e6f0"),
        xaxis=dict(gridcolor="#2a2f42"),
        yaxis=dict(gridcolor="#2a2f42", title="Monthly Rent (₹)"),
        margin=dict(t=20, b=0),
        showlegend=False,
        height=280,
    )
    return fig


def make_yield_gauge(yield_pct, city_avg):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=yield_pct,
        number={"suffix": "%", "font": {"size": 28, "color": "#e4e6f0"}},
        delta={"reference": city_avg, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 8], "tickcolor": "#7f8699"},
            "bar": {"color": "#6c63ff"},
            "bgcolor": "#1e2230",
            "bordercolor": "#2a2f42",
            "steps": [
                {"range": [0, 2.5], "color": "#3d1010"},
                {"range": [2.5, 4.5], "color": "#1e2230"},
                {"range": [4.5, 8],   "color": "#1a3d30"},
            ],
            "threshold": {"line": {"color": "#00c9a7", "width": 3}, "value": city_avg},
        },
        title={"text": "Gross Rental Yield", "font": {"color": "#7f8699", "size": 13}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e4e6f0",
        height=220, margin=dict(t=30, b=0, l=20, r=20)
    )
    return fig


def make_model_comparison_chart(pred):
    models = ["Linear Regression", "Random Forest", "Ensemble"]
    values = [
        pred.get("Linear Regression", 0),
        pred.get("Random Forest", 0),
        pred.get("Ensemble", pred.get("Rule-Based", 0)),
    ]
    colors = ["#f7b731", "#00c9a7", "#6c63ff"]

    fig = go.Figure(go.Bar(
        x=models, y=values, marker_color=colors,
        text=[f"₹{v:,.0f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e4e6f0"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#2a2f42", title="Monthly Rent (₹)"),
        margin=dict(t=20, b=0), height=260, showlegend=False,
    )
    return fig


with st.sidebar:
    st.markdown("## 🏙️ **RealAdvisor AI**")
    st.markdown('<p style="color:#7f8699;font-size:0.82rem;margin-top:-10px;">Agentic Real Estate Intelligence</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🏠 Property Details")

    city = st.selectbox("City", VALID_CITIES, index=0)

    locations = CITY_LOCATIONS.get(city, [])
    location  = st.selectbox("Location / Neighbourhood", locations)

    col1, col2 = st.columns(2)
    with col1:
        rooms     = st.number_input("Rooms (BHK)", min_value=1, max_value=6, value=2)
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)

    col3, col4 = st.columns(2)
    with col3:
        balconies = st.number_input("Balconies", min_value=0, max_value=4, value=1)
    with col4:
        size_sqft = st.number_input("Size (sq ft)", min_value=200, max_value=5000, value=1000, step=50)

    property_type = st.selectbox("Property Type", VALID_PROPERTY_TYPES)
    status        = st.selectbox("Furnishing Status", VALID_STATUSES)

    col5, col6 = st.columns(2)
    with col5:
        security_deposit = st.number_input("Security Deposit (₹)", min_value=0, max_value=500000, value=25000, step=5000)
    with col6:
        is_negotiable = st.checkbox("Negotiable", value=False)

    st.divider()
    st.markdown("### 👤 Investor Profile")

    purpose   = st.selectbox("Purpose", ["investment", "self-use"])
    risk      = st.selectbox("Risk Appetite", ["low", "moderate", "high"], index=1)
    horizon   = st.selectbox("Investment Horizon", ["short", "medium", "long"], index=1)
    budget    = st.slider("Budget (₹ Lakhs)", 20, 500, 60, step=5)
    exp_yield = st.slider("Expected Yield (%)", 2.0, 8.0, 4.0, step=0.5)

    st.divider()
    run_btn = st.button("🚀 Run Advisory Analysis")


# Header
st.markdown("""
<div style="margin-bottom:2rem;">
  <h1 style="margin:0;font-size:2.1rem;">Agentic Real Estate Advisory</h1>
  <p style="color:#7f8699;margin-top:4px;">Powered by LangGraph · Groq AI · FAISS RAG · Scikit-learn ML</p>
</div>
""", unsafe_allow_html=True)

# Architecture diagram
with st.expander("📐 Agent Workflow Architecture", expanded=False):
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    LANGGRAPH AGENTIC PIPELINE                       │
    │                                                                     │
    │  [Input] ──► [1.Validate] ──► [2.Predict] ──► [3.RAG Retrieve]    │
    │                                                        │            │
    │                                               [4.Comparables]      │
    │                                                        │            │
    │                                             [5.Risk Assess] ◄─ LLM │
    │                                                        │            │
    │                                           [6.Gen Advice] ◄──── LLM │
    │                                                        │            │
    │                                          [7.Report] ◄──────── LLM  │
    │                                                        │            │
    │                                                    [OUTPUT]         │
    └─────────────────────────────────────────────────────────────────────┘

    Components:
    • LangGraph StateGraph   — Explicit state management across 7 nodes
    • Groq (LLaMA 3.3 70B)  — Risk analysis, investment advice, report generation
    • FAISS + SentenceTransformers — Market knowledge RAG (no OpenAI needed)
    • Scikit-learn ML        — Price prediction (Linear Regression + Random Forest)
    • Streamlit              — Interactive web UI
    ```
    """)

#  Run the agent 
if run_btn:
    property_details = {
        "city":             city,
        "location":         location,
        "size_sqft":        size_sqft,
        "rooms":            rooms,
        "bathrooms":        bathrooms,
        "balconies":        balconies,
        "bhk_flag":         1,
        "status":           status,
        "property_type":    property_type,
        "is_negotiable":    1 if is_negotiable else 0,
        "security_deposit": security_deposit,
        "latitude":         CITY_COORDS[city]["lat"],
        "longitude":        CITY_COORDS[city]["lon"],
    }
    user_preferences = {
        "purpose":            purpose,
        "risk_appetite":      risk,
        "investment_horizon": horizon,
        "budget_lakhs":       budget,
        "expected_yield_pct": exp_yield,
    }

    #  Live progress 
    progress_placeholder = st.empty()
    step_placeholder     = st.empty()

    steps_done = 0
    progress_bar = progress_placeholder.progress(0, text="Initializing advisory pipeline...")

    def update_progress(step_idx: int, label: str):
        pct = int((step_idx / len(STEP_LABELS)) * 100)
        progress_placeholder.progress(pct, text=f"Step {step_idx}/{len(STEP_LABELS)}: {label}")
        step_placeholder.markdown(step_tracker_html(step_idx - 1), unsafe_allow_html=True)

    update_progress(1, "Validating property input...")

    from agent_graph import run_advisory

    # Stream step updates while running
    with st.spinner(""):
        result = run_advisory(property_details, user_preferences)

    progress_placeholder.progress(100, text="✅ Analysis complete!")
    step_placeholder.markdown(step_tracker_html(len(STEP_LABELS)), unsafe_allow_html=True)
    time.sleep(0.5)
    progress_placeholder.empty()
    step_placeholder.empty()

    st.session_state["result"] = result


if "result" in st.session_state:
    result = st.session_state["result"]
    pred   = result.get("prediction_result", {})
    an     = pred.get("analytics", {})
    props  = result.get("property_details", {})
    prefs  = result.get("user_preferences", {})
    comps  = result.get("comparables", [])
    risks  = result.get("risk_assessment", "")
    advice = result.get("investment_advice", "")
    report = result.get("final_report", "")
    logs   = result.get("step_logs", [])

    ens_rent = pred.get("Ensemble", pred.get("Rule-Based", 20000))
    rec_class, rec_label = get_rec_class(advice)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🏘️ Comparables", "⚠️ Risk & Advice", "📄 Full Report", "🔍 Agent Logs"
    ])

    with tab1:
        st.markdown(
            f'<div class="rec-badge {rec_class}">🎯 {rec_label}</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        tiles = [
            (format_inr(ens_rent),                             "Predicted Rent / Month"),
            (format_inr(an.get("annual_rent", ens_rent*12)),   "Annual Rental Income"),
            (f"{an.get('gross_yield_pct', '—')}%",             "Gross Yield"),
            (f"{an.get('price_to_rent_ratio', '—')}x",         "Price-to-Rent"),
            (f"{an.get('yoy_growth', '—')}%",                  "City YoY Growth"),
            (f"{an.get('vacancy_rate', '—')}%",                "Vacancy Rate"),
        ]
        for val, lbl in tiles:
            st.markdown(metric_tile(val, lbl), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Model Predictions Comparison")
            st.plotly_chart(make_model_comparison_chart(pred), use_container_width=True)
        with c2:
            st.markdown("#### Rental Yield vs City Average")
            st.plotly_chart(
                make_yield_gauge(
                    an.get("gross_yield_pct", 3.5),
                    an.get("city_avg_yield", 3.5)
                ),
                use_container_width=True
            )

        st.markdown(f"""
        <div class="card card-accent">
            <h4 style="margin:0 0 8px">🏠 Property Summary</h4>
            <b>{props.get('rooms')}BHK {props.get('property_type')}</b> in {props.get('location')}, {props.get('city')}<br>
            <span style="color:#7f8699">{props.get('size_sqft'):,.0f} sq ft · {props.get('status')} · 
            Security Deposit: ₹{props.get('security_deposit'):,} · 
            {'Negotiable' if props.get('is_negotiable') else 'Fixed'}</span>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🏘️ Comparable Property Analysis")
        if comps:
            st.plotly_chart(make_comparison_chart(comps, ens_rent), use_container_width=True)

            comp_df = pd.DataFrame(comps)[["name", "bhk", "size", "rent", "yield", "furnished", "similarity_score"]]
            comp_df.columns = ["Property", "BHK", "Size (sqft)", "Rent/Month (₹)", "Yield %", "Furnishing", "Match Score"]
            comp_df["Rent/Month (₹)"] = comp_df["Rent/Month (₹)"].apply(lambda x: f"₹{x:,}")
            comp_df["Match Score"]    = comp_df["Match Score"].apply(lambda x: f"{x:.0%}")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            avg_comp_rent   = np.mean([c["rent"] for c in comps])
            delta_pct       = ((ens_rent - avg_comp_rent) / avg_comp_rent) * 100
            delta_label     = "above" if delta_pct > 0 else "below"
            delta_color     = "#00c9a7" if delta_pct <= 5 else "#f7b731"
            st.markdown(f"""
            <div class="card card-success">
                <b>📈 Market Position:</b> Predicted rent of <b>{format_inr(ens_rent)}</b> is 
                <span style="color:{delta_color}">{abs(delta_pct):.1f}% {delta_label}</span> 
                the average comparable rent of ₹{avg_comp_rent:,.0f}.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No comparable properties found.")

    with tab3:
        st.markdown("### ⚠️ Risk Assessment")
        st.markdown(f'<div class="card card-warn">{risks}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### 💡 Investment Recommendation")
        rec_c, rec_l = get_rec_class(advice)
        st.markdown(f'<div class="rec-badge {rec_c}">🎯 {rec_l}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card card-accent">{advice}</div>', unsafe_allow_html=True)

        st.markdown("#### Risk Factor Radar")
        city_m = CITY_META.get(props.get("city", "Delhi"), {})
        cats   = ["Market Liquidity", "Yield Risk", "Vacancy Risk", "Regulatory", "Appreciation"]

        yield_risk    = max(0, (4.0 - an.get("gross_yield_pct", 3.5)) * 20)
        vacancy_risk  = an.get("vacancy_rate", 10) * 5
        liq_risk      = 60  # real estate always illiquid
        reg_risk      = 20  # RERA provides protection
        apprec_risk   = max(0, 80 - city_m.get("yoy_growth", 8) * 5)

        scores = [liq_risk, min(100, yield_risk), min(100, vacancy_risk), reg_risk, apprec_risk]
        scores.append(scores[0])
        cats.append(cats[0])

        fig_radar = go.Figure(go.Scatterpolar(
            r=scores, theta=cats, fill='toself',
            fillcolor="rgba(108,99,255,0.15)", line=dict(color="#6c63ff", width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1e2230",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2a2f42", tickcolor="#7f8699"),
                angularaxis=dict(gridcolor="#2a2f42")
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e4e6f0",
            height=350, margin=dict(t=30)
        )
        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_r2:
            st.markdown("""
            <div class="card" style="margin-top:2rem;">
                <b>Risk Scale</b><br>
                <span style="color:#00c9a7">0–30</span>: Low<br>
                <span style="color:#f7b731">31–60</span>: Moderate<br>
                <span style="color:#ee5a24">61–100</span>: High<br><br>
                <small style="color:#7f8699">Scores based on city metrics, yield delta, and market conditions.</small>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 📄 Full Investment Advisory Report")

        # Download button
        st.download_button(
            label="📥 Download Report (.md)",
            data=report,
            file_name=f"RealAdvisor_Report_{props.get('city')}_{props.get('rooms')}BHK.md",
            mime="text/markdown",
        )

        st.markdown("---")
        st.markdown(report)

        # Disclaimer box
        st.markdown("""
        <div class="card" style="border-color:#ee5a24;margin-top:2rem;">
            <b style="color:#ee5a24">⚖️ Legal & Financial Disclaimer</b><br>
            This advisory report is generated by an AI system and is provided for informational purposes only.
            It does not constitute financial, legal, or investment advice. Past market performance does not
            guarantee future returns. Always consult a SEBI-registered investment advisor and a qualified
            lawyer before making real estate decisions. Verify all property documents independently.
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.markdown("### 🔍 Agent Execution Logs")
        st.markdown("*Full trace of the LangGraph workflow execution*")

        for i, log in enumerate(logs):
            is_step = log.startswith(("🔍", "📊", "🗂️", "🏘️", "⚠️", "💡", "📝"))
            color = "#6c63ff" if is_step else "#7f8699"
            st.markdown(f"""
            <div style="background:#1e2230;border-left:3px solid {color};padding:8px 12px;
                        border-radius:0 6px 6px 0;margin-bottom:6px;font-size:0.85rem;
                        font-family:monospace;color:#e4e6f0;">
                {log}
            </div>
            """, unsafe_allow_html=True)

        # State summary
        with st.expander("📦 Raw Agent State (JSON)", expanded=False):
            safe_result = {k: v for k, v in result.items() if k not in ("step_logs",)}
            st.json(safe_result)

elif not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#7f8699;">
        <div style="font-size:4rem;margin-bottom:1rem;">🏙️</div>
        <h3 style="color:#e4e6f0;font-family:'Space Grotesk'">Ready to analyze your property</h3>
        <p>Configure property details in the sidebar and click <b>Run Advisory Analysis</b>.<br>
        The 7-step agentic pipeline will generate a comprehensive investment report.</p>
        <br>
        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
            <span style="background:#1e2230;padding:8px 16px;border-radius:20px;font-size:0.85rem;">🤖 LangGraph Workflow</span>
            <span style="background:#1e2230;padding:8px 16px;border-radius:20px;font-size:0.85rem;">⚡ Groq AI (LLaMA 3.3)</span>
            <span style="background:#1e2230;padding:8px 16px;border-radius:20px;font-size:0.85rem;">🗂️ FAISS RAG</span>
            <span style="background:#1e2230;padding:8px 16px;border-radius:20px;font-size:0.85rem;">📊 ML Price Models</span>
        </div>
    </div>
    """, unsafe_allow_html=True)