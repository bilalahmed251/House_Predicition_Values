## 📌 Project Overview
The goal of this project is to develop a predictive model that can estimate property values based on various demographic and geographic features. This project highlights the importance of data cleaning and scaling in achieving high model accuracy.

## 🧠 Key Techniques: Regression & Feature Engineering
To improve the model's performance and handle data irregularities, I implemented:

## Log Transformation: 
Applied to skewed features like total_rooms and population to normalize the distribution.
## Feature Scaling: 
Used StandardScaler to ensure all features contribute equally to the model's performance.
## Categorical Encoding: 
Handled ocean_proximity using One-Hot Encoding for machine readability.

## 🛠️ Technical Workflow

## Data Exploration & EDA: 
Analyzed correlations between features like median_income and median_house_value.
## Data Cleaning:
Handled missing values in the total_bedrooms column using median imputation to prevent data loss.
## Feature Engineering:
                       Created new meaningful attributes:
                       rooms_per_household: Average rooms per house.
                       bedroom_ratio: Proportion of bedrooms in a house.

## Model Training: 
Implemented Linear Regression and evaluated performance using the $R^2$ score.

## 💻 Tech Stack
## Python 
(Pandas, NumPy, Scikit-learn)
## Visualization: 
Matplotlib, Seaborn
## Environment: 
Jupyter Notebook / Google Colab

## 📂 Repository Structure
├── House Predict.ipynb    # Full analysis, cleaning, and modeling code
├── housing.csv            # California Housing dataset
├── model.pkl              # Saved Regression model (optional)
└── README.md              # Project documentation
