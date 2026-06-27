import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Establish clean structural UI defaults
st.set_page_config(
    page_title="Shopper Spectrum | Enterprise Intelligence Panel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render customized high-contrast CSS typography injections
st.markdown("""
    <style>
    .main-header {
        font-size: 32px; font-weight: 800; color: #0F172A; margin-bottom: 5px;
    }
    .sub-header {
        font-size: 15px; color: #475569; margin-bottom: 25px;
    }
    .metric-card-container {
        background-color: #FFFFFF; padding: 20px; border-radius: 12px;
        border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .rec-item {
        background: #F8FAFC; padding: 14px 20px; border-radius: 8px;
        margin-bottom: 10px; border-left: 4px solid #3B82F6;
        display: flex; justify-content: space-between; align-items: center;
    }
    .rec-title { font-weight: 600; color: #1E293B; font-size: 14px; }
    .rec-score { font-family: monospace; color: #2563EB; font-weight: 700; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def fetch_system_assets():
    try:
        with open('models/scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('models/kmeans.pkl', 'rb') as f: kmeans = pickle.load(f)
        with open('models/cluster_mapping.pkl', 'rb') as f: cluster_mapping = pickle.load(f)
        with open('models/item_similarity_df.pkl', 'rb') as f: item_sim = pickle.load(f)
        return scaler, kmeans, cluster_mapping, item_sim, True
    except FileNotFoundError:
        return None, None, None, None, False

scaler, kmeans, cluster_mapping, item_sim, status = fetch_system_assets()

# -----------------------------------------------------------------------------
# SIDEBAR CONSOLE MANAGEMENT ENVIRONMENT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🖥️ Core Telemetry Status")
    if status:
        st.success("● Pipeline Network Active")
        st.info(f"📦 Active Catalog SKU Volume: {len(item_sim.index):,}")
        st.info("🤖 Engine Configuration: K-Means v1.2")
    else:
        st.error("● Pipeline Assets Disconnected")
    
    st.markdown("---")
    st.markdown("### 📘 Model Parameters")
    st.caption("Clustering Model: Scikit-Learn KMeans\nDistance Metric: Transformed Euclidean\nRecommendation Metric: Pairwise Cosine Space")

# -----------------------------------------------------------------------------
# MAIN APP BODY ENVIRONMENT
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛒 Shopper Spectrum Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Corporate decision-support platform for dynamic user tier classification and algorithmic cross-sell recommendations.</div>', unsafe_allow_html=True)

if not status:
    st.error("🚨 Critical Backend Error: Unable to locate model binaries inside the `models/` directory. Run `build_models.py` to compile dependencies.")
else:
    tab1, tab2 = st.tabs(["🎯 Algorithmic Product Recommendations", "🔍 Real-time Customer Tiering Engine"])

    # =========================================================================
    # TAB 1: EXECUTIVE PRODUCT DISCOVERY LAYER
    # =========================================================================
    with tab1:
        st.markdown("### 📦 Inventory Affinity Exploration")
        st.write("Isolate systematic transaction combinations to evaluate related stock items across our entire retail operation.")
        
        catalog = sorted(list(item_sim.index))
        selected_item = st.selectbox("Search Core Product Catalog Inventory:", options=catalog)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Affinity Matches", type="primary"):
            if selected_item in item_sim.index:
                # Isolate matching values and pull indices 1 to 6 (ignoring self)
                targets = item_sim[selected_item].sort_values(ascending=False).iloc[1:6]
                
                st.markdown(f"#### 🏷️ Key Inventory Recommendations Matched with: *{selected_item}*")
                
                for index, (product_name, similarity_coefficient) in enumerate(targets.items(), start=1):
                    st.markdown(f"""
                        <div class="rec-item">
                            <span class="rec-title">{index}. {product_name}</span>
                            <span class="rec-score">Similarity: {similarity_coefficient:.4f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Selected SKU name not recognized inside operational data structures.")

    # =========================================================================
    # TAB 2: ADVANCED CUSTOMER BEHAVIOR ANALYSIS LAYER
    # =========================================================================
    with tab2:
        st.markdown("### 👤 Real-Time Vector Inference Engine")
        st.write("Input a consumer's behavior vector to accurately place them into an operational customer tier.")
        
        # Establish responsive grid input systems
        grid_col1, grid_col2, grid_col3 = st.columns(3)
        
        with grid_col1:
            in_recency = st.number_input("Recency (Days since last interaction activity)", min_value=1, max_value=365, value=14)
        with grid_col2:
            in_frequency = st.number_input("Frequency (Total aggregated historical invoices)", min_value=1, max_value=1500, value=22)
        with grid_col3:
            in_monetary = st.number_input("Monetary (Total net historical gross spending value $)", min_value=1.0, max_value=1000000.0, value=3150.0, step=50.0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Segment Inference Classifier", type="primary"):
            # Format raw inputs into matrix configurations
            raw_features = pd.DataFrame([[in_recency, in_frequency, in_monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            
            # Map input elements through log1p scaling pipeline transformations
            scaled_features = scaler.transform(np.log1p(raw_features))
            
            predicted_index = kmeans.predict(scaled_features)[0]
            assigned_label = cluster_mapping.get(predicted_index, "Occasional")
            
            # Determine visual indicator colors
            color_palettes = {"High-Value": "#16A34A", "Regular": "#2563EB", "Occasional": "#EA580C", "At-Risk": "#DC2626"}
            ui_color = color_palettes.get(assigned_label, "#2563EB")
            
            playbook_actions = {
                "High-Value": "💎 VIP Segment: Channel into high-tier loyalty pathways, offer premium first-access product drops, and assign a dedicated support desk.",
                "Regular": "📊 Mainstream Revenue Base: Utilize logic bundling, cross-category coupons, and volume incentives to scale basket sizes.",
                "Occasional": "🌱 Incipient Accounts: Use re-engagement email cadences and target them with item-affinity recommendations to build purchase habits.",
                "At-Risk": "⚠️ Churn Prevention Warning: Deploy strong re-activation offers, survey-incentives, and aggressive retention win-back discounts."
            }
            action_plan = playbook_actions.get(assigned_label, "")
            
            st.markdown("---")
            st.markdown("### 📋 Classification Executive Summary")
            
            # Display summary metric layouts
            ui_col1, ui_col2 = st.columns([1, 2])
            with ui_col1:
                st.markdown(f"""
                    <div class="metric-card-container" style="border-top: 6px solid {ui_color};">
                        <span style="font-size: 12px; text-transform: uppercase; color: #64748B; font-weight: 700;">Assigned Profile Group</span>
                        <h2 style="margin: 5px 0; color: {ui_color}; font-size: 26px;">{assigned_label}</h2>
                        <span style="font-size: 13px; color: #475569;">System Index: Cluster #{predicted_index}</span>
                    </div>
                """, unsafe_allow_html=True)
            with ui_col2:
                st.markdown(f"""
                    <div class="metric-card-container">
                        <span style="font-size: 12px; text-transform: uppercase; color: #64748B; font-weight: 700;">Operational Action Playbook</span>
                        <p style="margin-top: 10px; color: #1E293B; font-size: 14px; line-height: 1.6;">{action_plan}</p>
                    </div>
                """, unsafe_allow_html=True)