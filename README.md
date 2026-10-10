# 🎯 Lead Intent & Priority Scorer

### AI-Powered Lead Scoring & Conversion Prediction Platform

<p align="center">
  <img src="team_orion.png" alt="Team Orion Logo" width="160">
</p>

<p align="center">
  <strong>Developed by Team Orion</strong>
  <br>
  An intelligent lead scoring platform that helps businesses identify high-potential leads, predict conversion likelihood, and prioritize sales opportunities using machine learning.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly">
</p>

---

## 📑 Table of Contents

* [📌 Project Overview](#-project-overview)
* [🎯 Problem Statement](#-problem-statement)
* [✨ Key Features](#-key-features)
* [🛠️ Technology Stack](#️-technology-stack)
* [🧠 Machine Learning Workflow](#-machine-learning-workflow)
* [📂 Project Structure](#-project-structure)
* [⚙️ Installation and Setup](#️-installation-and-setup)
* [📊 Model Evaluation](#-model-evaluation)
* [💼 Business Benefits](#-business-benefits)
* [🚀 Future Enhancements](#-future-enhancements)
* [👥 Team Members](#-team-members)

---

## 📌 Project Overview

**Lead Intent & Priority Scorer** is a machine learning-powered web application designed to help businesses make smarter, data-driven decisions about their potential customers.

The platform analyzes lead attributes and generates conversion predictions to help sales teams identify promising prospects and prioritize follow-up activities.

Built using **Python, Scikit-learn, Pandas, Plotly, and Streamlit**, the application combines predictive analytics with an interactive dashboard.

The model is trained using the **X Education Lead Scoring dataset**.

### 🎯 Project Objectives

| Objective                 | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| 🎯 Lead Scoring           | Estimate the likelihood of a lead converting into a customer. |
| ⚡ Lead Prioritization     | Help sales teams focus on potentially valuable prospects.     |
| 📊 Data Analytics         | Identify patterns and trends in historical lead data.         |
| 📂 Bulk Prediction        | Score multiple leads through a CSV upload.                    |
| 📈 Performance Monitoring | Present model evaluation metrics and training information.    |

---

## 🎯 Problem Statement

Businesses often receive a large number of leads through marketing campaigns, websites, and other channels. However, sales teams have limited time and resources to engage with every lead equally.

Without a systematic prioritization process, businesses may spend valuable time on leads with lower conversion potential while overlooking more promising opportunities.

**Our solution:** Use machine learning to estimate lead conversion probability and provide an interactive platform for analyzing and prioritizing leads.

---

## ✨ Key Features

| Feature                        | Description                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------- |
| 📊 **Executive Overview**      | Displays key performance indicators and an overview of lead conversion insights. |
| 📈 **Lead Analytics**          | Explores lead characteristics, conversion patterns, and analytical insights.     |
| 🎯 **Individual Lead Scoring** | Accepts lead information and generates a conversion prediction.                  |
| 📂 **Bulk Lead Scoring**       | Processes multiple leads from an uploaded CSV file.                              |
| 🤖 **Predictive Analytics**    | Uses a trained machine learning pipeline to estimate conversion likelihood.      |
| ⚡ **Lead Prioritization**      | Helps organize leads into priority categories using predicted scores.            |
| 📉 **Model Performance**       | Displays model evaluation metrics and training information.                      |
| 🎨 **Interactive Dashboard**   | Provides a user-friendly interface with data visualizations.                     |

---

## 🛠️ Technology Stack

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| **Python**       | Core programming language                 |
| **Pandas**       | Data manipulation and preprocessing       |
| **NumPy**        | Numerical operations                      |
| **Scikit-learn** | Machine learning and predictive modeling  |
| **Streamlit**    | Interactive web application               |
| **Plotly**       | Interactive charts and data visualization |
| **Joblib**       | Saving and loading the trained model      |
| **CSV & JSON**   | Dataset storage and model metadata        |
| **Git & GitHub** | Version control and project hosting       |

---

## 🧠 Machine Learning Workflow

The application follows a structured machine learning pipeline.

```text
       ┌─────────────────────────┐
       │  Lead Scoring Dataset   │
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │  Data Preprocessing     │
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Feature Preparation     │
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Model Training          │
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Model Evaluation        │
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Lead Conversion Scoring │
       └────────────┬────────────┘
                    ▼
       ┌─────────────────────────┐
       │ Streamlit Dashboard     │
       └─────────────────────────┘
```

### Workflow Explanation

1. **Data Collection:** Load historical lead information.
2. **Data Preprocessing:** Prepare data for model training and prediction.
3. **Feature Preparation:** Process input variables into the format expected by the model.
4. **Model Training:** Train a machine learning pipeline using historical data.
5. **Model Evaluation:** Evaluate predictive performance using a test dataset.
6. **Lead Scoring:** Generate conversion predictions for individual or multiple leads.
7. **Visualization:** Display analytics and model information through the dashboard.

---

## 📂 Project Structure

```text
lead-intent-priority-scorer/
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

| File / Directory                      | Description                                                     |
| ------------------------------------- | --------------------------------------------------------------- |
| `app.py`                              | Main Streamlit dashboard and application logic                  |
| `train.py`                            | Model training and evaluation script                            |
| `requirements.txt`                    | Python project dependencies                                     |
| `team_orion.png`                      | Team Orion logo                                                 |
| `data/raw/Leads.csv`                  | Historical lead dataset                                         |
| `models/lead_scoring_pipeline.joblib` | Saved machine learning pipeline                                 |
| `models/metrics.json`                 | Stored model evaluation and training information                |
| `models/feature_names.json`           | Model input feature information                                 |
| `models/model_comparison.csv`         | Model comparison results, if generated by the training workflow |

---

## ⚙️ Installation and Setup

Follow these steps to run the project on your local machine.

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

**Windows PowerShell**

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

The application should open automatically in your browser.

If it does not, open the local address displayed in your terminal, usually:

`http://localhost:8501`

### System Requirements

* Python installed on your system
* Git installed on your system
* Internet access for cloning the repository and installing dependencies

---

## 📊 Model Evaluation

The application includes a **Model Performance** section for reviewing model evaluation results and training information.

| Evaluation Metric | Purpose                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------- |
| **Accuracy**      | Measures the proportion of correct predictions.                                             |
| **Precision**     | Measures how many predicted positive cases are actually positive.                           |
| **Recall**        | Measures how many actual positive cases the model identifies.                               |
| **F1-Score**      | Combines precision and recall into a single metric.                                         |
| **ROC-AUC**       | Measures how well the model distinguishes between classes across classification thresholds. |

The dashboard can also display:

* Training record count
* Testing record count
* Number of input features
* Historical conversion rate

Refer to the application's **Model Performance** page and generated model metrics for actual results.

---

## 💼 Business Benefits

| Business Challenge                         | Potential Benefit                                                           |
| ------------------------------------------ | --------------------------------------------------------------------------- |
| Too many leads to review manually          | Automated lead scoring helps streamline prioritization.                     |
| Limited sales team resources               | Sales teams can prioritize follow-up using predicted conversion likelihood. |
| Difficulty identifying conversion patterns | Analytics can help reveal trends in historical lead data.                   |
| Need to evaluate multiple leads            | Bulk scoring supports batch prediction workflows.                           |
| Limited visibility into model performance  | Evaluation metrics provide insight into predictive performance.             |

---

## 🚀 Future Enhancements

* [ ] Deploy the application to a cloud platform.
* [ ] Add model explainability using SHAP.
* [ ] Improve lead-source and campaign performance analysis.
* [ ] Introduce historical prediction tracking.
* [ ] Integrate with CRM platforms.
* [ ] Add configurable lead-priority thresholds.
* [ ] Improve model performance through experimentation and feature engineering.
* [ ] Add automated model monitoring and retraining.

---

## 👥 Team Members

**🔷 TEAM ORION**

| Sr. No. | Team Member            |
| :-----: | ---------------------- |
|    1    | **Yashraj Kad**        |
|    2    | **Aditya Borgaonkar**  |
|    3    | **Anish Jagtap**       |
|    4    | **Harshwardhan Chate** |


### 📌 Project Details

| Detail                   | Information                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Project Name**         | Lead Intent & Priority Scorer                                                                          |
| **Team Name**            | Team Orion                                                                                             |
| **Domain**               | Artificial Intelligence, Machine Learning & Data Analytics                                             |
| **Programming Language** | Python                                                                                                 |
| **Framework**            | Streamlit                                                                                              |
| **Repository**           | [Lead Intent & Priority Scorer on GitHub](https://github.com/aditya280505/lead-intent-priority-scorer) |

---

<p align="center">
  <strong>Built with Python, Machine Learning & Streamlit</strong>
  <br>
  <strong>💙 Team Orion</strong>
</p>
