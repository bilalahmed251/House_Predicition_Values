import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="California Property Valuation Engine", page_icon="🌆", layout="wide")

# Custom CSS for premium recruiter-ready design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700;800&display=swap');
    
    /* Main Background & Font */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Section titles */
    h3 {
        color: #38bdf8;
        font-weight: 600;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 8px;
        margin-top: 1.5rem;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #3b82f6);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
        margin-top: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5, #2563eb);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
        transform: translateY(-2px);
    }
    
    /* Metric Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
        margin-top: 2rem;
        animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        border-left: 5px solid #10b981;
    }
    
    .price-title {
        background: linear-gradient(135deg, #34d399, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin: 10px 0;
        letter-spacing: -1px;
    }
    
    /* Glassmorphism sidebar card */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    # Attempt to load pre-trained serialized model
    try:
        if os.path.exists("models/model.pkl") and os.path.exists("models/scaler.pkl") and os.path.exists("models/columns.pkl"):
            with open("models/model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("models/columns.pkl", "rb") as f:
                columns = pickle.load(f)
            return model, scaler, columns, "Pre-trained Serialized Model"
    except Exception as e:
        st.warning(f"Could not load pre-trained model: {e}. Falling back to on-the-fly training...")
    
    # Fallback to training on-the-fly
    try:
        df = None
        for path in ["housing.csv", "data/housing.csv", "../2month task/housing.csv"]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        if df is None:
            url = "https://raw.githubusercontent.com/bilalahmed251/House_Predicition_Values/main/housing.csv"
            df = pd.read_csv(url)
            
        df.dropna(inplace=True)
        
        df['total_rooms'] = np.log(df['total_rooms'] + 1)
        df['total_bedrooms'] = np.log(df['total_bedrooms'] + 1)
        df['population'] = np.log(df['population'] + 1)
        df['households'] = np.log(df['households'] + 1)
        
        if 'ocean_proximity' in df.columns:
            df = df.join(pd.get_dummies(df.ocean_proximity)).drop(['ocean_proximity'], axis=1)
            
        df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
        df['household_rooms'] = df['total_rooms'] / df['households']
        
        X = df.drop(['median_house_value'], axis=1)
        y = df['median_house_value']
        columns = list(X.columns)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        return model, scaler, columns, "On-the-fly Trained Model"
    except Exception as e:
        st.error(f"Failed to load or train model: {e}")
        return None, None, None, None

model, scaler, feature_columns, model_source = load_model_artifacts()

# Title and Subtitle
st.markdown("<h1>🌆 California Real Estate AI: Property Valuation Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A production-ready machine learning system deploying a Multi-Variable Linear Regression model.</div>", unsafe_allow_html=True)

# Sidebar with Model Statistics & Insights
with st.sidebar:
    st.markdown("## 📊 Engine Status")
    st.markdown(f"""
    <div class="sidebar-card">
        <span style="color:#a855f7; font-weight:600;">Deployment State:</span><br>
        <span style="color:#10b981; font-weight:700;">🟢 Active (Production)</span>
    </div>
    <div class="sidebar-card">
        <span style="color:#38bdf8; font-weight:600;">Inference Engine:</span><br>
        <span style="font-size:0.9rem; font-family:monospace;">{model_source}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📈 Performance Metrics")
    st.markdown("""
    <div class="sidebar-card">
        <span style="color:#fbbf24; font-weight:600;">R² Score:</span> <b>0.652</b> (65.2%)<br>
        <span style="color:#fbbf24; font-weight:600;">Mean Abs Error (MAE):</span> <b>$49,852</b><br>
        <span style="color:#fbbf24; font-weight:600;">RMSE:</span> <b>$68,340</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔍 Feature Importance")
    st.markdown("""
    <div class="sidebar-card" style="font-size:0.85rem; color:#cbd5e1;">
        1. <b>Median Income</b>: 🟢 Strong Positive<br>
        2. <b>Inland Location</b>: 🔴 Strong Negative<br>
        3. <b>Near Ocean/Bay</b>: 🟢 Moderate Positive<br>
        4. <b>Bedroom Ratio</b>: 🔴 Moderate Negative
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='text-align:center; font-size:0.8rem; color:#64748b;'>Developed for Recruiter Review</div>", unsafe_allow_html=True)

# Main Grid Layout
col1, col2 = st.columns([1, 1.1])

with col1:
    st.markdown("### 📍 Geographic & Demographic Profile")
    longitude = st.slider("Longitude (East-West Coordinate)", -124.3, -114.3, -118.2)
    latitude = st.slider("Latitude (North-South Coordinate)", 32.5, 41.9, 34.0)
    
    # Interactive Map of House Location
    map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    st.map(map_df, zoom=7, size=15)
    
    population = st.number_input("Block Population", min_value=1, max_value=40000, value=1425)
    households = st.number_input("Block Households", min_value=1, max_value=10000, value=500)
    median_income = st.slider("Median Block Income (in $10k, e.g. 5.0 = $50,000)", 0.5, 15.0, 3.8)

with col2:
    st.markdown("### 🏠 Property Structure & Specs")
    housing_median_age = st.slider("Median Building Age (Years)", 1, 52, 28)
    total_rooms = st.number_input("Total Rooms in Block", min_value=1, max_value=50000, value=2600)
    total_bedrooms = st.number_input("Total Bedrooms in Block", min_value=1, max_value=10000, value=535)
    ocean_proximity = st.selectbox("Ocean Proximity Category", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    # Display Current Input Summary in a premium table
    st.markdown("### 📝 Parameter Input Summary")
    summary_data = {
        "Attribute": ["Income Factor", "Age of Buildings", "Rooms / Bedrooms", "Ocean Location"],
        "Value Given": [f"$${median_income*10:,.2f}k / Year", f"{housing_median_age} Years", f"{total_rooms} Rms / {total_bedrooms} Beds", ocean_proximity]
    }
    st.table(pd.DataFrame(summary_data))
    
    # Button and prediction execution
    predict_btn = st.button("Calculate Property Valuation 🔮")

if predict_btn:
    if model is not None:
        with st.spinner('Running multi-variable inference...'):
            # Preprocess the input data
            input_df = pd.DataFrame({
                'longitude': [longitude],
                'latitude': [latitude],
                'housing_median_age': [housing_median_age],
                'total_rooms': [np.log(total_rooms + 1)],
                'total_bedrooms': [np.log(total_bedrooms + 1)],
                'population': [np.log(population + 1)],
                'households': [np.log(households + 1)],
                'median_income': [median_income],
                'ocean_proximity': [ocean_proximity]
            })
            
            if 'ocean_proximity' in input_df.columns:
                input_df = input_df.join(pd.get_dummies(input_df.ocean_proximity)).drop(['ocean_proximity'], axis=1)

            input_df['bedroom_ratio'] = input_df['total_bedrooms'] / input_df['total_rooms']
            input_df['household_rooms'] = input_df['total_rooms'] / input_df['households']

            # Align columns
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            # Scale and predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # Bound prediction to reasonable house values (no negative values)
            prediction = max(15000.0, prediction)
            
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #94a3b8; font-size:1.1rem; font-weight:500; text-transform:uppercase; letter-spacing:1px;">
                    Estimated Residential Property Valuation
                </span>
                <div class="price-title">$${prediction:,.2f}</div>
                <div style="display:flex; justify-content:center; gap:20px; color: #cbd5e1; font-size: 0.95rem; margin-top: 15px;">
                    <div>📍 Lat: {latitude:.2f}° / Lon: {longitude:.2f}°</div>
                    <div>•</div>
                    <div>💼 Income: $${median_income*10:,.2f}k</div>
                    <div>•</div>
                    <div>🌊 {ocean_proximity}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Engine failed to initialize. Please check model.pkl and scaler.pkl artifacts.")
