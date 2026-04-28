"""
app.py — PropPredict AI Streamlit App (Redesigned — Luxury Blue Theme)
Run: streamlit run app/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go

from models.predict import predict_property, top_properties
from chatbot.chatbot import PropertyChatbot

st.set_page_config(
    page_title="PropPredict AI — Airbnb Property Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LUXURY BLUE THEME ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #050d2e 0%, #0a1854 45%, #0e2270 100%) !important;
    border-right: 1px solid rgba(99,160,255,0.15) !important;
}
[data-testid="stSidebar"] * { color: #c8d8ff !important; }
[data-testid="stSidebar"] .stRadio label { color: #c8d8ff !important; font-size: 14px !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #8aaeff !important; font-size: 12px !important; }
[data-testid="stSidebar"] hr { border-color: rgba(99,160,255,0.2) !important; }
[data-testid="stSidebar"] h1 { color: #ffffff !important; font-family: 'Nunito', sans-serif !important; font-size: 22px !important; font-weight: 700 !important; }

[data-testid="stAppViewContainer"] > .main { background: #f0f4ff !important; }
.main .block-container { background: #f0f4ff !important; padding-top: 2rem !important; }

h1 { font-family: 'Nunito', sans-serif !important; color: #050d2e !important; font-weight: 600 !important; }
h2, h3 { font-family: 'Nunito', sans-serif !important; color: #0a1854 !important; font-weight: 700 !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0a1854 0%, #1a3080 100%) !important;
    border-radius: 16px !important; padding: 20px !important;
    border: 1px solid rgba(99,160,255,0.25) !important;
    box-shadow: 0 4px 24px rgba(10,24,84,0.18) !important;
}
[data-testid="stMetric"] label { color: #8aaeff !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Nunito', sans-serif !important; font-weight: 600 !important; font-size: 24px !important; }

.stButton > button {
    background: linear-gradient(135deg, #1651e8 0%, #0a35c4 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    padding: 10px 20px !important; box-shadow: 0 4px 14px rgba(22,81,232,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(22,81,232,0.5) !important; transform: translateY(-1px) !important; }

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #ffffff !important; border: 1.5px solid rgba(22,81,232,0.25) !important; border-radius: 10px !important;
}
.stSelectbox label, .stMultiSelect label, .stSlider label { font-weight: 500 !important; color: #0a1854 !important; font-size: 13px !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(22,81,232,0.08) !important; border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important; border: 1px solid rgba(22,81,232,0.15) !important;
}
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; color: #3050a0 !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1651e8, #0a35c4) !important; color: #ffffff !important; }

[data-testid="stDataFrame"] { border-radius: 16px !important; border: 1px solid rgba(22,81,232,0.15) !important; overflow: hidden !important; }

.stProgress > div > div > div > div { background: linear-gradient(90deg, #1651e8, #63d0a0) !important; border-radius: 8px !important; }
.stProgress > div > div > div { background: rgba(22,81,232,0.15) !important; border-radius: 8px !important; }

[data-testid="stChatMessage"] { border-radius: 16px !important; border: 1px solid rgba(22,81,232,0.12) !important; }
[data-testid="stChatInputTextArea"] { border: 1.5px solid rgba(22,81,232,0.3) !important; border-radius: 12px !important; background: #ffffff !important; }

[data-testid="stForm"] {
    background: #ffffff !important; border: 1.5px solid rgba(22,81,232,0.15) !important;
    border-radius: 20px !important; padding: 24px !important; box-shadow: 0 4px 20px rgba(10,24,84,0.08) !important;
}

[data-testid="stCaptionContainer"] p { color: #5570a0 !important; }
hr { border-color: rgba(22,81,232,0.15) !important; }

[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio > div > label {
    background: rgba(99,160,255,0.08) !important; border-radius: 10px !important;
    padding: 10px 14px !important; width: 100% !important;
    transition: all 0.15s !important; border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover { background: rgba(99,160,255,0.18) !important; border-color: rgba(99,160,255,0.3) !important; }
[data-testid="stSidebar"] [aria-checked="true"] { background: rgba(22,81,232,0.35) !important; border-color: rgba(99,160,255,0.5) !important; }

.stat-band {
    background: linear-gradient(135deg, #050d2e 0%, #0e2270 100%);
    border-radius: 20px; padding: 28px 32px; margin-bottom: 24px;
    display: flex; align-items: center; justify-content: space-between;
    border: 1px solid rgba(99,160,255,0.15); box-shadow: 0 8px 32px rgba(5,13,46,0.25);
}
.stat-band-title { font-family: 'Nunito', sans-serif; font-size: 22px; font-weight: 600; color: #ffffff; letter-spacing: 0px; line-height: 1.25; }
.stat-band-sub { font-size: 13px; color: #8aaeff; margin-top: 4px; }
.badge-live { display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:30px;font-size:12px;font-weight:600;background:rgba(99,208,160,0.2);color:#63d0a0;border:1px solid rgba(99,208,160,0.4); }
.section-card { background: #ffffff; border-radius: 20px; padding: 24px; border: 1px solid rgba(22,81,232,0.12); box-shadow: 0 2px 16px rgba(10,24,84,0.07); margin-bottom: 20px; }
.section-card h3 { font-family: 'Nunito', sans-serif !important; font-size: 15px !important; font-weight: 700 !important; color: #050d2e !important; margin-bottom: 14px !important; }
.signal-chip { display:inline-block;padding:5px 14px;border-radius:30px;font-size:12px;font-weight:700;letter-spacing:0.5px; }
.signal-strong-buy { background:#d0f0e0;color:#065f3c; }
.signal-buy { background:#dceeff;color:#0a3585; }
.signal-hold { background:#fff4cc;color:#7a5500; }
.signal-avoid { background:#ffe0e0;color:#7a1010; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/listings_featured.csv")

@st.cache_data
def load_metrics():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"models","metrics.json")
    with open(path) as f: return json.load(f)

df = load_data()
metrics = load_metrics()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(240,244,255,0.5)",
    font=dict(family="DM Sans, sans-serif", color="#050d2e"),
    margin=dict(t=30, b=20, l=10, r=10),
    hoverlabel=dict(bgcolor="#050d2e", font_color="#ffffff", font_family="DM Sans"),
    xaxis=dict(gridcolor="rgba(22,81,232,0.08)", linecolor="rgba(22,81,232,0.2)"),
    yaxis=dict(gridcolor="rgba(22,81,232,0.08)", linecolor="rgba(22,81,232,0.2)"),
)
CITY_COLORS = {"Dubai":"#1651e8","Goa":"#0db870","Maldives":"#e8a216"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px 0">
      <div style="font-size:32px;margin-bottom:8px">🏡</div>
      <div style="font-family:'Nunito',sans-serif;font-size:20px;font-weight:600;color:#ffffff;letter-spacing:0px">PropPredict AI</div>
      <div style="font-size:12px;color:#8aaeff;margin-top:6px">Airbnb Property Intelligence</div>
      <div style="margin-top:8px">
        <span style="color:#0db870;font-size:11px">● Dubai</span>
        <span style="color:#e8a216;font-size:11px;margin-left:8px">● Goa</span>
        <span style="color:#63a0ff;font-size:11px;margin-left:8px">● Maldives</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", ["🏠 Dashboard","🤖 AI Chatbot","🔮 Predict Property","🗺 Market Map","📊 Analytics","📁 About Project"])
    st.divider()
    m = metrics
    st.markdown("""<div style="font-size:10px;color:#8aaeff;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:10px">Model Performance</div>""", unsafe_allow_html=True)
    for label, value in [("Price R²", m['price_predictor']['R2']),
                          ("ROI Accuracy", m['roi_classifier']['accuracy']),
                          (f"Score MAE", f"±{m['investment_scorer']['MAE_pts']} pts")]:
        st.markdown(f"""
        <div style="background:rgba(99,160,255,0.1);border-radius:10px;padding:10px 14px;
                    border:1px solid rgba(99,160,255,0.2);margin-bottom:8px">
          <div style="font-size:10px;color:#8aaeff;text-transform:uppercase;letter-spacing:0.8px">{label}</div>
          <div style="font-size:20px;font-weight:700;color:#ffffff;font-family:'Nunito',sans-serif">{value}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown(f"""
    <div class="stat-band">
      <div>
        <div class="stat-band-title">Airbnb Property<br>Intelligence Platform</div>
        <div class="stat-band-sub">Real-time investment signals · Dubai · Goa · Maldives</div>
      </div>
      <div style="text-align:right">
        <div class="badge-live">● LIVE DATA</div>
        <div style="color:#8aaeff;font-size:12px;margin-top:8px">{len(df):,} listings analysed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Listings", f"{len(df):,}")
    c2.metric("Avg Nightly Price", f"${df['nightly_price_usd'].mean():.0f}")
    c3.metric("Avg ROI", f"{df['roi_pct'].mean():.1f}%")
    c4.metric("Avg Occupancy", f"{df['occupancy_rate'].mean():.0%}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card"><h3>📦 Price Distribution by City</h3>', unsafe_allow_html=True)
        fig = px.box(df, x="city", y="nightly_price_usd", color="city",
                     color_discrete_map=CITY_COLORS, labels={"nightly_price_usd":"Nightly Price (USD)","city":""})
        fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><h3>💰 ROI % by Area — Top 12</h3>', unsafe_allow_html=True)
        top_areas = df.groupby(["city","area"])["roi_pct"].mean().reset_index().sort_values("roi_pct", ascending=False).head(12)
        fig2 = px.bar(top_areas, x="roi_pct", y="area", color="city", orientation="h",
                      color_discrete_map=CITY_COLORS, labels={"roi_pct":"ROI %","area":""})
        fig2.update_layout(**{**PLOTLY_LAYOUT, "yaxis": dict(autorange="reversed", gridcolor="rgba(22,81,232,0.08)", linecolor="rgba(22,81,232,0.2)")})
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>📊 Investment Signal Distribution</h3>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    for city, col in zip(["Dubai","Goa","Maldives"],[col3,col4,col5]):
        sub = df[df["city"]==city]["investment_label"].value_counts()
        fig3 = px.pie(values=sub.values, names=sub.index, title=city,
                      color_discrete_sequence=["#1651e8","#0db870","#e8a216","#e84040"], hole=0.45)
        fig3.update_layout(**{**PLOTLY_LAYOUT, "margin":dict(t=50,b=10,l=10,r=10)})
        col.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Property":
    st.markdown("""
    <div class="stat-band">
      <div><div class="stat-band-title">Property Prediction<br>Engine</div>
      <div class="stat-band-sub">Input details → ML-powered price, ROI & investment score</div></div>
      <div style="font-size:40px">🔮</div>
    </div>""", unsafe_allow_html=True)

    with st.form("predict_form"):
        col1,col2,col3 = st.columns(3)
        city      = col1.selectbox("🌍 City", ["Dubai","Goa","Maldives"])
        bedrooms  = col2.slider("🛏 Bedrooms", 0, 5, 2)
        bathrooms = col3.slider("🚿 Bathrooms", 1, 5, 2)
        col4,col5,col6 = st.columns(3)
        review_score = col4.slider("⭐ Review Score", 3.0, 5.0, 4.5, 0.1)
        dist_beach   = col5.slider("🏖 Beach Distance (km)", 0.0, 10.0, 1.5, 0.1)
        dist_airport = col6.slider("✈️ Airport Distance (km)", 1.0, 50.0, 15.0, 1.0)
        col7,col8 = st.columns(2)
        ptype = col7.selectbox("🏠 Property Type", ["Apartment","Villa","Studio","Penthouse","Cottage","Overwater Bungalow"])
        amenities_sel = col8.multiselect("✨ Amenities", ["Pool","WiFi","AC","Parking","Gym","Garden","Sea View","Butler Service"], default=["WiFi","AC","Parking"])
        submitted = st.form_submit_button("🔮 Run Prediction", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Analysing with ML models..."):
            result = predict_property({"city":city,"bedrooms":bedrooms,"bathrooms":bathrooms,
                "review_score":review_score,"distance_to_beach_km":dist_beach,
                "distance_to_airport_km":dist_airport,"property_type":ptype,
                "amenities":"|".join(amenities_sel),"num_reviews":50})
        sig = result["investment_label"]
        sig_class = {"Strong Buy":"signal-strong-buy","Buy":"signal-buy","Hold":"signal-hold","Avoid":"signal-avoid"}.get(sig,"signal-hold")
        st.markdown(f"""
        <div class="stat-band" style="margin-top:20px">
          <div>
            <div style="font-size:11px;color:#8aaeff;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Prediction Result</div>
            <div class="stat-band-title">{city} · {ptype}</div>
            <div style="margin-top:12px"><span class="signal-chip {sig_class}">{sig}</span></div>
          </div>
          <div style="text-align:right">
            <div style="font-size:42px;font-weight:600;color:#fff;font-family:'Nunito',sans-serif;line-height:1">{result['investment_score']}</div>
            <div style="font-size:13px;color:#8aaeff">Investment Score / 100</div>
          </div>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Nightly Price", f"${result['predicted_nightly_price_usd']:,}")
        c2.metric("Annual Revenue", f"${result['predicted_annual_revenue_usd']:,}")
        c3.metric("Estimated ROI", f"{result['estimated_roi_pct']}%")
        c4.metric("Property Value", f"${result['estimated_property_price_usd']:,}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(result['investment_score'] / 100)
        st.caption(f"📊 {result['explanation']}")

        st.markdown(f'<div class="section-card"><h3>Top Properties in {city}</h3>', unsafe_allow_html=True)
        top_df = top_properties(city=city, min_roi=5, top_n=8)
        st.dataframe(top_df.style.background_gradient(subset=["investment_score"], cmap="Blues"),
                     use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":
    st.markdown("""
    <div class="stat-band">
      <div><div class="stat-band-title">PropPredict AI<br>Chatbot</div>
      <div class="stat-band-sub">Ask anything about Airbnb properties in Dubai, Goa & Maldives</div></div>
      <div style="font-size:40px">🤖</div>
    </div>""", unsafe_allow_html=True)

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = PropertyChatbot()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"assistant","content":"Hi! I'm **PropPredict AI** 🏡\n\nI can help you with:\n- 💰 Property price predictions\n- 📈 ROI & investment signals\n- 🗺 Best areas in Dubai, Goa & Maldives\n- 🏠 Property type comparisons\n\nWhat would you like to know?"}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    st.divider()
    st.markdown("**Quick questions:**")
    cols = st.columns(4)
    quick = ["Best area in Dubai under $500K?","Goa beachfront villa ROI?","Maldives overwater bungalow price?","Compare all 3 markets"]
    for i,(col,q) in enumerate(zip(cols,quick)):
        if col.button(q, key=f"qp{i}", use_container_width=True):
            st.session_state._quick_prompt = q

    user_input = st.chat_input("Ask about properties, ROI, prices, investment strategy...")
    if hasattr(st.session_state, "_quick_prompt"):
        user_input = st.session_state._quick_prompt
        del st.session_state._quick_prompt

    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.chat_message("user", avatar="👤"): st.markdown(user_input)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analysing market data..."):
                reply = st.session_state.chatbot.chat(user_input)
            st.markdown(reply)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = [{"role":"assistant","content":"Chat reset! Ask me anything about Dubai, Goa or Maldives properties 🏡"}]
        st.session_state.chatbot.reset()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MARKET MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 Market Map":
    st.markdown("""
    <div class="stat-band">
      <div><div class="stat-band-title">Interactive<br>Property Map</div>
      <div class="stat-band-sub">Explore listings by investment score & nightly price</div></div>
      <div style="font-size:40px">🗺</div>
    </div>""", unsafe_allow_html=True)

    city_filter = st.selectbox("Filter by city", ["All","Dubai","Goa","Maldives"])
    map_df = df if city_filter=="All" else df[df["city"]==city_filter]
    map_df = map_df.sample(min(300, len(map_df)))

    fig = px.scatter_mapbox(map_df, lat="latitude", lon="longitude",
        color="investment_score", size="nightly_price_usd", hover_name="id",
        hover_data={"area":True,"property_type":True,"nightly_price_usd":True,"roi_pct":True,"investment_label":True,"latitude":False,"longitude":False},
        color_continuous_scale=[[0,"#e84040"],[0.5,"#e8a216"],[1,"#0db870"]],
        mapbox_style="carto-darkmatter", zoom=3, height=560,
        labels={"investment_score":"Score","nightly_price_usd":"$/night"})
    fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown("""
    <div class="stat-band">
      <div><div class="stat-band-title">Deep Analytics</div>
      <div class="stat-band-sub">Feature correlations, ROI heatmaps & occupancy trends</div></div>
      <div style="font-size:40px">📊</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Price vs Features","🔥 ROI Heatmap","📅 Occupancy Trends"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        col_x = st.selectbox("Select feature for X axis", ["bedrooms","review_score","distance_to_beach_km","num_reviews","amenity_count"])
        fig = px.scatter(df, x=col_x, y="nightly_price_usd", color="city", trendline="ols",
                         color_discrete_map=CITY_COLORS, opacity=0.55, labels={"nightly_price_usd":"Nightly Price (USD)"})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        pivot = df.pivot_table(values="roi_pct", index="property_type", columns="city", aggfunc="mean")
        fig2 = px.imshow(pivot.round(1), color_continuous_scale=[[0,"#0a1854"],[0.5,"#1651e8"],[1,"#63d0a0"]],
                         text_auto=True, aspect="auto", title="Avg ROI % by Property Type & City")
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        occ_data = df.groupby(["city","property_type"])["occupancy_rate"].mean().reset_index()
        fig3 = px.bar(occ_data, x="property_type", y="occupancy_rate", color="city", barmode="group",
                      color_discrete_map=CITY_COLORS, labels={"occupancy_rate":"Avg Occupancy Rate","property_type":"Property Type"})
        fig3.update_layout(**PLOTLY_LAYOUT)
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 About Project":
    st.markdown("""
    <div class="stat-band">
      <div><div class="stat-band-title">About This<br>Project</div>
      <div class="stat-band-sub">Architecture, tech stack & real-world business value</div></div>
      <div style="font-size:40px">📁</div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown('<div class="section-card"><h3>🏗 Architecture</h3>', unsafe_allow_html=True)
        arch_data = {"Layer":["Data collection","Feature engineering","Price prediction","ROI classification","Investment scoring","AI chatbot","Dashboard","Deployment"],
                     "Technology":["Python, Pandas, Inside Airbnb CSVs","Scikit-learn, NumPy","XGBoost (R² = 0.91)","Random Forest (99.7% accuracy)","XGBoost regression","Claude API + rule-based fallback","Streamlit + Plotly","Streamlit Cloud / Hugging Face Spaces"]}
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><h3>📊 ML Model Performance</h3>', unsafe_allow_html=True)
        m = load_metrics()
        st.metric("Price Predictor R²", m["price_predictor"]["R2"])
        st.metric("Price MAE", f"${m['price_predictor']['MAE_usd']}")
        st.metric("ROI Accuracy", m["roi_classifier"]["accuracy"])
        st.metric("Score MAE", f"±{m['investment_scorer']['MAE_pts']} pts")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>💼 Real-world Business Value</h3>', unsafe_allow_html=True)
    b1,b2,b3,b4 = st.columns(4)
    for col,emoji,title,desc in zip([b1,b2,b3,b4],["🏙","🌴","🐠","✈️"],
        ["Dubai Investors","Goa Villa Owners","Maldives Resorts","Travelers"],
        ["Identify highest-ROI zones before purchase","Dynamic pricing to maximise peak season revenue",
         "Occupancy forecasting 30–90 days ahead","Find best value-for-money properties by area"]):
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a1854,#1a3080);border-radius:14px;
                    padding:16px;text-align:center;border:1px solid rgba(99,160,255,0.2)">
          <div style="font-size:28px;margin-bottom:8px">{emoji}</div>
          <div style="font-family:'Nunito',sans-serif;font-size:13px;font-weight:700;color:#ffffff;margin-bottom:6px">{title}</div>
          <div style="font-size:11px;color:#8aaeff;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>📂 Data Sources</h3>', unsafe_allow_html=True)
    st.markdown("""
- [Inside Airbnb](https://insideairbnb.com) — real listing & review data
- [Dubai Pulse / DLD](https://www.dubaipulse.gov.ae) — government property transactions
- [Kaggle Airbnb datasets](https://www.kaggle.com)
- Synthetic data for Maldives/Goa (calibrated to real market distributions)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
