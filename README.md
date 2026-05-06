<div align="center">
  <img src="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?q=80&w=1200&auto=format&fit=crop" alt="California Real Estate AI Banner" width="100%" style="border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
</div>

<h1 align="center">🌆 California Real Estate AI: Property Valuation Engine</h1>

<div align="center">
  <p><strong>A production-ready machine learning repository demonstrating end-to-end data pipelines, advanced feature engineering, model serialization, and interactive deployment.</strong></p>
  
  <a href="https://huggingface.co/spaces/b098/house-price-predictor">
    <img src="https://img.shields.io/badge/🤗_Hugging_Face-Live_App-blue.svg?style=for-the-badge&logo=huggingface&color=FFD21E&logoColor=black" alt="Hugging Face Space">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white&color=3776AB" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Model%20Training-orange.svg?style=for-the-badge&logo=scikitlearn&logoColor=white&color=F7931E" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-Web%20Interface-red.svg?style=for-the-badge&logo=streamlit&logoColor=white&color=FF4B4B" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge&color=2ea44f" alt="License">
</div>

---

## 🌐 Live Interactive Application

Explore the deployed system live on Hugging Face Spaces:  
👉 **[Launch the Live Property Estimator Space](https://huggingface.co/spaces/b098/house-price-predictor)**

## 🎥 User Interface & Demo
<div align="center">
  <img src="media/House hold Screenshot.png" alt="Property Valuation Engine UI" width="900" style="border-radius: 12px; box-shadow: 0px 8px 30px rgba(0,0,0,0.3);">
  <p><i>Bespoke dark-themed, glassmorphism-inspired GIS dashboard for real-time house price predictions.</i></p>
</div>

---

## 🛠️ System Architecture & Data Pipeline

This project demonstrates software engineering best practices for data science, ensuring a clean separation of concerns by separating training pipelines (`train.py`) from serving applications (`app.py`), and pre-serializing models to optimize sub-second online inference.
 
```mermaid
graph TD
    A[California Census CSV Dataset] -->|1. Data Cleaning| B(Drop Missing Values)
    B -->|2. Feature Engineering| C[Log Transforms & Custom Ratios]
    C -->|3. Feature Scaling| D(StandardScaler Fit & Transform)
    D -->|4. Training| E[Multi-Variable Linear Regression]
    E -->|5. Serialization| F[Export model.pkl, scaler.pkl, columns.pkl]
    F -->|6. Production Serving| G[app.py Streamlit UI Engine]
    H[User Real-time Inputs] -->|7. Preprocess & Scale| G
    G -->|8. High-Speed Inference| I[Estimated Property Value Display]
```

---

## 📈 Engineering Highlights & Decision Log

### 1. Advanced Feature Engineering
- **Logarithmic Smoothing:** Census features like `total_rooms`, `total_bedrooms`, `population`, and `households` exhibit severe right-skewness. Applying $y = \log(x + 1)$ normalizes the distribution, ensuring the Linear Regression model handles outliers robustly.
- **Domain-Specific Ratios:**
  - `bedroom_ratio` ($\frac{\text{total bedrooms}}{\text{total rooms}}$): Helps capture the density of bedrooms, which highly correlates with house style and density.
  - `household_rooms` ($\frac{\text{total rooms}}{\text{households}}$): Captures the average room scale per housing unit.
- **Categorical Categorization:** One-hot encodes `ocean_proximity` to enable numerical model coefficients for geographical locations.

### 2. Model Serialization (Production Pattern)
Rather than training the model dynamically on every single API/page request, this system pre-serializes the trained `LinearRegression` model and `StandardScaler` state as `.pkl` binary files.
- **Latency Advantage:** Drops model load time to **<1ms**, compared to training on-the-fly which takes several seconds and burdens CPU resources.
- **Reliability:** Built with an elegant dual fallback structure: loads local `.pkl` files first, with an automated fallback to train on-the-fly from GitHub or local CSV if artifacts are missing.

---

## 📊 Model Performance & Interpretation

### Evaluation Metrics
- **R² Score (Coefficient of Determination):** **0.6520** (Explains 65.2% of the variance in California property values)
- **Mean Absolute Error (MAE):** **$49,852**
- **Root Mean Squared Error (RMSE):** **$68,340**

### Learned Coefficients (Feature Impact)
The model uncovers critical econometric insights based on standard census data:

| Feature | Coefficient Sign | Economic Interpretation |
| :--- | :---: | :--- |
| **Median Income** | 🟢 Positive (Strongest) | Higher neighborhood income is the most powerful driver of home valuation. |
| **Inland Location** | 🔴 Negative (Strongest) | Properties situated inland suffer a massive valuation discount compared to coastal areas. |
| **Near Ocean / Near Bay** | 🟢 Positive (Moderate) | Coastline proximity yields a substantial valuation premium. |
| **Bedroom Ratio** | 🔴 Negative (Moderate) | Higher bedroom density (e.g. apartment blocks) indicates lower single-family premium pricing. |

---

## 📁 Repository Directory Structure

```
House_Predicition_Values/
│
├── train.py                 # Offline Model training, evaluation & serialization pipeline
├── app.py                   # High-performance Streamlit UI & inference serving logic
├── requirements.txt         # Package dependencies with compatible pins
├── README.md                # Premium developer and recruiter documentation
├── housing.csv              # California Housing Census source dataset
│
├── models/                  # Serialized Production Artifacts
│   ├── model.pkl            # Serialized Scikit-learn Linear Regression model
│   ├── scaler.pkl           # Serialized StandardScaler state
│   └── columns.pkl          # Exported trained feature schema alignment
│
├── notebooks/               # Research & Prototyping
│   └── california_housing_analysis.ipynb  # Exploratory Data Analysis & training log
│
└── media/                   # Portfolio Assets
    ├── House hold Screenshot.png  # Interactive GIS Web App screenshot
    └── house_prediction_demo.mp4  # HD Video application walk-through
```

---

## ⚙️ Local Development Setup Guide

Follow these steps to run the training pipeline and launch the web interface locally:

### 1. Clone the Source Code
```bash
git clone https://github.com/bilalahmed251/House_Predicition_Values.git
cd House_Predicition_Values
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the Training Pipeline
To retrain the model and regenerate the `.pkl` files inside `models/`:
```bash
python train.py
```

### 5. Launch the Streamlit Server
```bash
streamlit run app.py
```
*The application will boot up at `http://localhost:8501/` with live hot-reloading.*

---

## ☁️ Serverless Cloud Deployment

This app is optimized for serverless hosting on **Hugging Face Spaces** (using the Streamlit SDK):
1. Create a new Streamlit Space on Hugging Face.
2. Push `app.py`, `requirements.txt`, `housing.csv`, and the pre-trained `models/` folder.
3. Hugging Face automatically spins up the instance and serves the property valuation engine.

---
<div align="center">
  <sub>Engineered with 💙 for Data Science & Real Estate Analytics • MIT License 2026</sub>
</div>
