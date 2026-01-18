import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Configure Streamlit FIRST
# ============================================================
st.set_page_config(page_title="Engine RUL Prediction", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# LSTM Model Definition (Must be defined before loading model)
# ============================================================
class RULLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.regressor(last_hidden).squeeze(1)

# ============================================================
# Load Configuration, Model and Data
# ============================================================
@st.cache_resource
def load_model_and_data():
    with open("app_config.json", "r") as f:
        config = json.load(f)
    model = pickle.load(open("lstm_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    predictions_df = pd.read_csv("predictions.csv")
    return config, model, scaler, predictions_df

CONFIG, model, scaler, predictions_df = load_model_and_data()

# ============================================================
# Utility Functions
# ============================================================
def get_engine_prediction(engine_id):
    """Get prediction for a specific engine"""
    if engine_id < 1 or engine_id > len(predictions_df):
        return None
    return predictions_df.iloc[engine_id - 1].to_dict()

def get_all_predictions():
    """Get all engine predictions"""
    return predictions_df.to_dict(orient="records")

def get_health_status_color(health_percentage):
    """Return color based on health percentage"""
    if health_percentage >= 75:
        return "🟢"  # Green - Good
    elif health_percentage >= 50:
        return "🟡"  # Yellow - Fair
    elif health_percentage >= 25:
        return "🟠"  # Orange - Poor
    else:
        return "🔴"  # Red - Critical

# ============================================================
# Streamlit UI
# ============================================================

# ------------------------------------------------------------
# 1. Custom CSS for "Clean & Smooth" Look
# ------------------------------------------------------------
st.markdown("""
    <style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card-like containers for metrics */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-title {
        color: #555;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Health Status Colors */
    .health-good { color: #2ecc71; }
    .health-fair { color: #f1c40f; }
    .health-poor { color: #e67e22; }
    .health-critical { color: #e74c3c; font-weight: bold; }

    /* Custom Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Urgent Alert Box */
    .urgent-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 4px;
        color: #991b1b;
        margin-bottom: 20px;
    }
    .urgent-title {
        font-weight: 700;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 2. Helper Functions for UI Logic
# ------------------------------------------------------------
def get_status_badge(health_pct):
    if health_pct >= 75:
        return "🟢 Normal"
    elif health_pct >= 50:
        return "🟡 Warning"
    elif health_pct >= 25:
        return "🟠 Action Needed"
    else:
        return "🔴 Critical"

def calculate_time_to_maintenance(rul, cycles_per_day=5):
    """Estimate days remaining based on RUL and assumed usage."""
    days = rul / cycles_per_day
    return days

# ------------------------------------------------------------
# 3. Sidebar Navigation & Global Settings
# ------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208728.png", width=60)
    st.title("Factory Ops")
    st.caption("Predictive Maintenance Tool")
    
    st.markdown("---")
    
    page = st.radio("Navigate", ["Analysis Dashboard", "Engine Inspector", "Full Fleet Data"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    cycles_per_day = st.number_input("Est. Cycles per Day", min_value=1, value=5, help="Used to calculate days until repair")
    st.caption("Adjust this to match factory throughput.")

# ------------------------------------------------------------
# 4. Main Views
# ------------------------------------------------------------

if page == "Analysis Dashboard":
    st.title("🏭 Plant Maintenance Overview")
    st.markdown("Real-time health monitoring of all active engines.")
    
    # 4.1 Top Level Metrics
    total_engines = len(predictions_df)
    avg_rul = predictions_df["Predicted_RUL"].mean()
    critical_count = len(predictions_df[predictions_df["Health_Percentage"] < 25])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-title">Active Engines</div><div class="metric-value">{total_engines}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-title">Avg Fleet RUL</div><div class="metric-value">{avg_rul:.0f} <span style="font-size:1rem">cycles</span></div></div>', unsafe_allow_html=True)
    
    # Highlight critical engines in red if any
    crit_color = "#e74c3c" if critical_count > 0 else "#2ecc71"
    c3.markdown(f'<div class="metric-card" style="border-bottom: 3px solid {crit_color}"><div class="metric-title">Critical Attention</div><div class="metric-value" style="color:{crit_color}">{critical_count}</div></div>', unsafe_allow_html=True)
    
    fleet_health = predictions_df["Health_Percentage"].mean()
    c4.markdown(f'<div class="metric-card"><div class="metric-title">Fleet Health</div><div class="metric-value">{fleet_health:.1f}%</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 4.2 Priority Maintenance Queue
    st.subheader("🛠️ Maintenance Priority Queue")
    
    # Filter for urgency
    urgent_df = predictions_df.sort_values("Predicted_RUL").head(10).copy()
    urgent_df["Est. Days Remaining"] = urgent_df["Predicted_RUL"].apply(lambda x: f"{x/cycles_per_day:.1f} days")
    urgent_df["Status"] = urgent_df["Health_Percentage"].apply(get_status_badge)
    
    # Display as a styled dataframe with selective columns
    st.dataframe(
        urgent_df[["Engine_ID", "Status", "Predicted_RUL", "Est. Days Remaining", "Health_Percentage"]].style.background_gradient(subset=["Health_Percentage"], cmap="RdYlGn", vmin=0, vmax=100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Engine_ID": "Engine #",
            "Predicted_RUL": "Cycles Left",
            "Health_Percentage": st.column_config.ProgressColumn("Health", format="%d%%", min_value=0, max_value=100)
        }
    )
    
    # 4.3 Visualizations
    st.markdown("### 📊 Fleet Distribution")
    colA, colB = st.columns(2)
    with colA:
        fig = px.histogram(predictions_df, x="Predicted_RUL", nbins=30, color_discrete_sequence=['#3498db'])
        fig.update_layout(title="RUL Distribution (Cycles)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        # Pie chart of health status
        status_counts = predictions_df["Health_Percentage"].apply(lambda x: 
            "Critical" if x < 25 else "Poor" if x < 50 else "Fair" if x < 75 else "Good"
        ).value_counts()
        fig2 = px.pie(values=status_counts.values, names=status_counts.index, 
                      color=status_counts.index,
                      color_discrete_map={"Critical": "#e74c3c", "Poor": "#e67e22", "Fair": "#f1c40f", "Good": "#2ecc71"})
        fig2.update_layout(title="Fleet Health Segmentation", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Engine Inspector":
    st.title("🔍 Deep Dive Analysis")
    
    col_sel, col_blank = st.columns([1, 2])
    with col_sel:
        search_id = st.number_input("Enter Engine ID to Inspect:", min_value=1, max_value=len(predictions_df), value=1)
    
    engine_data = get_engine_prediction(search_id)
    
    if engine_data:
        rul = engine_data['Predicted_RUL']
        health = engine_data['Health_Percentage']
        days_left = calculate_time_to_maintenance(rul, cycles_per_day)
        
        # 5.1 Urgency Alert
        if health < 25:
             st.markdown(f"""
                <div class="urgent-box">
                    <div class="urgent-title">⚠️ CRITICAL MAINTENANCE REQUIRED</div>
                    This engine has less than 25% health remaining. Immediate repair scheduling is recommended to avoid failure.<br>
                    Estimated <b>{days_left:.1f} days</b> of operation remaining at current capacity.
                </div>
            """, unsafe_allow_html=True)
        elif health < 50:
            st.info(f"💡 Maintenance recommended within {days_left:.1f} days. Monitor closely.")
        
        # 5.2 Key Stats Layout
        col1, col2, col3 = st.columns(3)
        
        # Gauge Chart
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Health Score"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                    'bar': {'color': "#2c3e50"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 25], 'color': '#ffcccc'},
                        {'range': [25, 50], 'color': '#ffe6cc'},
                        {'range': [50, 75], 'color': '#ffffcc'},
                        {'range': [75, 100], 'color': '#ccffcc'}],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col1:
            st.markdown("### Prediction")
            st.metric("Remaining Useful Life", f"{rul:.1f} Cycles")
            st.metric("Estimated Time Left", f"{days_left:.1f} Days")
            st.caption(f"Based on {cycles_per_day} cycles/day")
            
        with col3:
            st.markdown("### Sensor Snapshot")
            # Display a mini table of sensor readings (random subset for demo/visual)
            # In a real app we might show trends. For now, showing current values.
            st.write("Current key sensor readings:")
            sensor_data = {k:v for k,v in engine_data.items() if k not in ['Engine_ID', 'Predicted_RUL', 'Health_Percentage']}
            st.json(sensor_data, expanded=False)

elif page == "Full Fleet Data":
    st.title("📋 Full Fleet Register")
    
    st.dataframe(
        predictions_df.style.background_gradient(subset=["Health_Percentage"], cmap="RdYlGn", vmin=0, vmax=100),
        use_container_width=True,
        column_order=["Engine_ID", "Predicted_RUL", "Health_Percentage"] + [c for c in predictions_df.columns if c not in ["Engine_ID", "Predicted_RUL", "Health_Percentage"]],
        hide_index=True
    )

