# 🏠 Real Estate Investment Advisor  
### Predicting Property Profitability & Future Value using Machine Learning

---

# 📌 Project Overview

The **Real Estate Investment Advisor** is a Machine Learning based analytical system developed to assist investors, property buyers, and real estate companies in making intelligent and data-driven investment decisions.

The system predicts:

✅ Whether a property is a **Good Investment**  
✅ Estimated **Future Property Price after 5 Years**

The project combines:
- Data Analysis
- Feature Engineering
- Machine Learning
- Model Evaluation
- Streamlit Deployment
- MLflow Experiment Tracking

to create a complete real estate analytics solution.

---

# 🎯 Objectives

The main objectives of this project are:

- To analyze real estate property data
- To predict profitable investment opportunities
- To forecast future property prices
- To reduce financial risks in property investment
- To automate real estate investment analysis
- To provide real-time prediction using Machine Learning

---

# 🚀 Features

✅ Data Preprocessing & Cleaning  
✅ Exploratory Data Analysis (EDA)  
✅ Feature Engineering  
✅ Classification Model  
✅ Regression Model  
✅ Future Property Price Prediction  
✅ Investment Recommendation System  
✅ Streamlit Web Application  
✅ MLflow Experiment Tracking  
✅ Model Saving using Pickle  

---

# 🧠 Machine Learning Models Used

## 1️⃣ Classification Model
Used to predict whether a property is a good investment.

### Algorithm:
- Random Forest Classifier

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 2️⃣ Regression Model
Used to predict future property price after 5 years.

### Algorithm:
- Random Forest Regressor

### Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Analysis |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Streamlit | Frontend Deployment |
| MLflow | Experiment Tracking |
| Matplotlib | Data Visualization |
| Seaborn | Data Visualization |
| Pickle | Model Saving |

---

# 📂 Project Structure

```bash
Real_Estate_Investment_Advisor/
│
├── app/
│   └── app.py
│
├── data/
│   ├── india_housing_prices.csv
│   └── cleaned_real_estate.csv
│
├── models/
│   ├── classification_model.pkl
│   └── regression_model.pkl
│
├── mlruns/
│
├── src/
│   ├── data_preprocessing.py
│   └── model_training.py
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ System Workflow

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Model Training
        ↓
Model Evaluation
        ↓
MLflow Experiment Tracking
        ↓
Streamlit Deployment
        ↓
Real-Time Predictions
```

---

# 📊 Dataset Features

The dataset contains important property-related attributes such as:

- State
- City
- Property Type
- BHK
- Size in SqFt
- Price in Lakhs
- Year Built
- Nearby Schools
- Nearby Hospitals
- Parking Space
- Amenities
- Furnished Status

---

# 🧹 Data Preprocessing

The following preprocessing techniques were applied:

- Handling Missing Values
- Removing Duplicate Records
- Feature Engineering
- Label Encoding
- Data Cleaning
- Derived Feature Creation

### Features Created:
- Price_per_SqFt
- Age_of_Property
- Future_Price_5Y
- Good_Investment

---

# 📈 Feature Engineering

The project uses feature engineering to improve prediction accuracy.

### Formula Used for Future Price Prediction

:contentReference[oaicite:0]{index=0}

Where:
- \( r \) = Growth Rate
- \( t \) = Time in Years

---

# 📉 Exploratory Data Analysis (EDA)

EDA was performed using:
- Histograms
- Heatmaps
- Correlation Analysis
- Price Distribution Graphs
- BHK Distribution
- Price vs Size Analysis

---

# 🤖 MLflow Integration

MLflow is integrated for:

- Experiment Tracking
- Metric Logging
- Parameter Logging
- Model Management
- Run Comparison

### Metrics Tracked:
- Accuracy
- Precision
- Recall
- F1 Score
- MAE
- RMSE
- R² Score

---

# 🌐 Streamlit Application

An interactive Streamlit web application was developed for users to:

- Enter property details
- Predict investment profitability
- Estimate future property price
- Get instant real-time results

---

# ▶️ How to Run the Project

---

## 1️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run Data Preprocessing

```bash
python src/data_preprocessing.py
```

---

## 3️⃣ Train Machine Learning Models

```bash
python src/model_training.py
```

---

## 4️⃣ Start Streamlit Application

```bash
streamlit run app/app.py
```

---

## 5️⃣ Start MLflow UI

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 📋 Results

The project successfully:

✅ Predicted profitable investment opportunities  
✅ Forecasted future property prices  
✅ Integrated MLflow for tracking  
✅ Developed Streamlit frontend  
✅ Improved decision-making using machine learning  

---

# 📌 Advantages

- Data-driven investment recommendations
- Faster property analysis
- Reduced financial risk
- Automated prediction system
- Real-time prediction interface
- Scalable analytical solution

---

# ⚠️ Limitations

- Predictions depend on dataset quality
- Market conditions may change unpredictably
- Economic fluctuations can affect accuracy

---

# 🔮 Future Enhancements

- Live Real Estate API Integration
- Cloud Deployment
- GIS Map Visualization
- Deep Learning Models
- Mobile Application
- Real-Time Market Analytics

---

# 🎓 Academic Relevance

This project demonstrates practical implementation of:

- Machine Learning
- Predictive Analytics
- Data Science
- Web Deployment
- Experiment Tracking
- Real Estate Analytics

It is suitable for:
- Major Projects
- Research Work
- Academic Demonstrations
- Machine Learning Case Studies

---

# 👨‍💻 Author

Developed for academic and research purposes using Machine Learning and Real Estate Analytics.

---

# 📚 References

- Python Documentation
- Scikit-learn Documentation
- Streamlit Documentation
- MLflow Documentation
- Machine Learning Research Papers
- Real Estate Analytics Studies

---