# 🎯 Lead Intent & Priority Scorer

### AI-Powered Lead Scoring and Conversion Prediction Platform

<p align="center">
  <img src="team_orion.png" alt="Team Orion Logo" width="180">
</p>

<p align="center">
  <b>Developed by Team Orion</b>
</p>

<p align="center">
  A machine learning-powered web application that predicts lead conversion probability and helps businesses prioritize potential customers.
</p>

---

## 📌 Project Overview

The **Lead Intent & Priority Scorer** is an AI-powered analytics platform designed to help businesses identify promising leads and prioritize their sales efforts.

The application uses machine learning to analyze lead attributes, estimate conversion probability, and classify leads into priority categories. It also provides interactive dashboards and bulk lead scoring capabilities to support data-driven decision-making.

The project uses the X Education Lead Scoring dataset to train and evaluate its predictive model.

## 🎯 Problem Statement

Businesses often generate a large number of leads but have limited time and resources to follow up with every potential customer.

Without an effective prioritization system, sales teams may spend time on leads that are less likely to convert while missing valuable opportunities.

This project addresses the problem by using machine learning to estimate lead conversion probability and help sales teams prioritize their follow-up activities.

## ✨ Key Features

* **📊 Executive Overview:** View key performance indicators and overall lead conversion insights.
* **📈 Lead Analytics:** Explore lead data and understand conversion patterns.
* **🎯 Individual Lead Scoring:** Enter lead information and generate a conversion prediction.
* **📂 Bulk Lead Scoring:** Upload a CSV file to score multiple leads.
* **🤖 Machine Learning Predictions:** Estimate the likelihood of a lead converting.
* **⚡ Lead Prioritization:** Organize leads into priority categories based on their predicted scores.
* **📉 Model Performance:** Review model evaluation metrics and training information.
* **🎨 Interactive Dashboard:** Explore insights through a user-friendly Streamlit interface.

## 🛠️ Tech Stack

| Category             | Technologies  |
| -------------------- | ------------- |
| Programming Language | Python        |
| Data Processing      | Pandas, NumPy |
| Machine Learning     | Scikit-learn  |
| Web Framework        | Streamlit     |
| Data Visualization   | Plotly        |
| Model Storage        | Joblib        |
| Data Format          | CSV, JSON     |
| Version Control      | Git, GitHub   |

## 🧠 Machine Learning Workflow

The project follows a machine learning workflow to generate lead conversion predictions.

1. **Data Collection:** Load the lead dataset.
2. **Data Preprocessing:** Prepare the data for model training.
3. **Feature Engineering:** Process input features for prediction.
4. **Model Training:** Train a machine learning pipeline using historical lead data.
5. **Model Evaluation:** Evaluate model performance on a test dataset.
6. **Lead Scoring:** Generate conversion predictions for new leads.
7. **Visualization:** Present predictions and analytics through the dashboard.

## 📂 Project Structure

```text
Lead-Intent-Priority-Scorer/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── team_orion.png
│
├── data/
│   └── raw/
│       └── Leads.csv
│
└── models/
    ├── lead_scoring_pipeline.joblib
    ├── metrics.json
    ├── feature_names.json
    └── model_comparison.csv
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/aditya280505/lead-intent-priority-scorer.git
```

### 2. Navigate to the Project Directory

```bash
cd lead-intent-priority-scorer
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, visit the local URL displayed in your terminal, usually `http://localhost:8501`.

## 📊 Model Evaluation

The project includes a model evaluation section that displays training information and performance metrics.

Evaluation can include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Training and testing record counts

The actual performance depends on the trained model and the dataset. Refer to the application's **Model Performance** section for the current evaluation results.

## 💡 Business Benefits

* Helps sales teams prioritize promising leads.
* Supports more efficient follow-up planning.
* Reduces reliance on manual lead assessment.
* Provides insights into historical conversion patterns.
* Enables scoring of individual leads and bulk datasets.
* Supports data-driven sales and marketing decisions.

## 🚀 Future Enhancements

* Deploy the application to a cloud platform.
* Add model explainability using SHAP.
* Introduce lead-source and campaign performance analysis.
* Add historical prediction tracking.
* Integrate CRM platforms.
* Improve model performance through experimentation and feature engineering.

## 👥 Team Orion

**Project:** Lead Intent & Priority Scorer
**Domain:** Artificial Intelligence, Machine Learning and Data Analytics
**Team:** Orion



---

<p align="center">
  <b>Built with Python, Machine Learning and Streamlit by Team Orion </b>
</p>
