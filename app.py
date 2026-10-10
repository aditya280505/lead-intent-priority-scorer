from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------------------------
# 1. PROJECT PATHS AND PAGE CONFIGURATION
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "leads.csv"
MODEL_PATH = BASE_DIR / "models" / "lead_scoring_pipeline.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.json"

st.set_page_config(
    page_title="Lead Intent & Priority Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)



# -------------------------------------------------
# 2. CUSTOM STYLING
# -------------------------------------------------

st.markdown("""
<style>

/* ================================================
   GLOBAL THEME
================================================ */

.stApp {
    background-color: #F4F7FB !important;
    color: #172033 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #F4F7FB !important;
}

[data-testid="stMain"] {
    background-color: transparent !important;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1500px !important;
}

/* ================================================
   TYPOGRAPHY
================================================ */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #172554 !important;
    font-weight: 700 !important;
}

.stApp p,
.stApp li,
.stApp label {
    color: #334155;
}

/* ================================================
   KPI METRIC CARDS
================================================ */

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #2563EB !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04) !important;
    transition: border-color 0.2s ease;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {
    color: #64748B !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #172554 !important;
    -webkit-text-fill-color: #172554 !important;
    font-size: 28px !important;
    font-weight: 750 !important;
}

/* ================================================
   INPUT LABELS
================================================ */

.stTextInput label,
.stNumberInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label,
.stDateInput label,
.stSlider label,
.stRadio label,
.stCheckbox label,
.stFileUploader label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* ================================================
   INPUT FIELDS
================================================ */

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stDateInput input {
    background-color: #FFFFFF !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] * {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

/* Dropdown options */
div[role="listbox"],
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #172033 !important;
}

li[role="option"] * {
    color: #172033 !important;
}

/* Checkbox */
div[data-testid="stCheckbox"] label {
    color: #334155 !important;
}

/* ================================================
   BUTTONS
================================================ */

.stButton button,
.stDownloadButton button,
div[data-testid="stFormSubmitButton"] button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: background-color 0.2s ease;
}

.stButton button *,
.stDownloadButton button *,
div[data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stButton button:hover,
.stDownloadButton button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

/* ================================================
   SIDEBAR
================================================ */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

section[data-testid="stSidebar"] * {
    color: #172033;
}

/* ================================================
   DATA TABLES
================================================ */

div[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ================================================
   ALERTS
================================================ */

div[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* ================================================
   EXPANDERS
================================================ */

div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] summary {
    color: #172033 !important;
}

/* ================================================
   HERO BANNER
================================================ */

.hero-box {
    background: linear-gradient(110deg, #172554, #2563EB) !important;
    color: #FFFFFF !important;
    padding: 28px !important;
    border-radius: 14px !important;
    margin-bottom: 22px !important;
}

.hero-box h1,
.hero-box h2,
.hero-box h3,
.hero-box p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* ================================================
   SMALL NOTES
================================================ */

.small-note {
    color: #64748B !important;
    font-size: 13px !important;
}

/* ================================================
   TRAINING INFORMATION JSON
================================================ */

div[data-testid="stJson"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

div[data-testid="stJson"] *,
div[data-testid="stJson"] pre,
div[data-testid="stJson"] code {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

/* ================================================
   FILE UPLOADER
================================================ */

div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px dashed #94A3B8 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)





# -------------------------------------------------
# 3. LOAD DATA AND MODEL
# -------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None

    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_feature_names():
    if not FEATURE_PATH.exists():
        return None

    with open(FEATURE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


df = load_data()
model = load_model()
metrics = load_metrics()
feature_names = load_feature_names()


# -------------------------------------------------
# 4. CHECK REQUIRED FILES
# -------------------------------------------------

if df is None:
    st.error("Dataset not found. Check data/raw/leads.csv.")
    st.stop()

if model is None:
    st.error("Trained model not found. Run python train.py first.")
    st.stop()

if "Converted" not in df.columns:
    st.error("The dataset does not contain the expected Converted column.")
    st.stop()

if feature_names is None:
    st.error("Feature names not found. Run python train.py again.")
    st.stop()



# -------------------------------------------------
# 5. SIDEBAR
# -------------------------------------------------

import os

# TEAM ORION LOGO
logo_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "team_orion.png"
)

if os.path.exists(logo_path):
    st.sidebar.image(
        logo_path,
        use_container_width=True
    )
else:
    st.sidebar.warning("Team Orion logo not found.")

st.sidebar.title("🎯 Lead Scoring")
st.sidebar.caption("Lead Intent & Priority Scorer")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Lead Analytics",
        "Score a Lead",
        "Bulk Lead Scoring",
        "Model Performance",
    ],
)







# -------------------------------------------------
# 6. HELPER FUNCTIONS
# -------------------------------------------------

def get_priority(probability):
    """
    Illustrative operational thresholds.
    These are not validated business cutoffs.
    """
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    return "Low"


def prepare_input(record):
    """
    Ensure input contains the same feature columns
    used during training.
    """
    prepared = pd.DataFrame([record])

    for column in feature_names:
        if column not in prepared.columns:
            prepared[column] = np.nan

    return prepared[feature_names]


def score_records(records):
    """
    Score multiple records using the saved pipeline.
    """
    clean_records = records.copy()
    clean_records.columns = clean_records.columns.str.strip()

    # Remove the outcome column if it appears in the input.
    clean_records = clean_records.drop(
        columns=["Converted"],
        errors="ignore",
    )

    # Add missing expected columns and remove extras.
    for column in feature_names:
        if column not in clean_records.columns:
            clean_records[column] = np.nan

    model_input = clean_records[feature_names]

    probabilities = model.predict_proba(model_input)[:, 1]

    output = clean_records.copy()
    output["Conversion Probability (%)"] = np.round(
        probabilities * 100, 2
    )
    output["Priority"] = [
        get_priority(float(probability))
        for probability in probabilities
    ]

    return output


def format_percentage(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


# -------------------------------------------------
# 7. EXECUTIVE OVERVIEW
# -------------------------------------------------

if page == "Executive Overview":

    st.markdown(
        """
        <div class="hero-box">
            <h1>🎯 Lead Intent & Priority Scorer</h1>
            <p>
                Machine Learning-powered lead analytics
                and conversion probability estimation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_leads = len(df)
    converted = int((df["Converted"] == 1).sum())
    not_converted = int((df["Converted"] == 0).sum())

    conversion_rate = (
        converted / total_leads if total_leads else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Leads", f"{total_leads:,}")
    col2.metric("Converted Leads", f"{converted:,}")
    col3.metric("Not Converted", f"{not_converted:,}")
    col4.metric("Historical Conversion Rate", format_percentage(conversion_rate))

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Historical Conversion Distribution")

        status_counts = (
            df["Converted"]
            .map({1: "Converted", 0: "Not Converted"})
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Lead Count")
        )

        fig = px.pie(
            status_counts,
            names="Status",
            values="Lead Count",
            hole=0.55,
            title="Converted vs Not Converted",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Lead Conversion Distribution")

        fig = px.bar(
            status_counts,
            x="Status",
            y="Lead Count",
            color="Status",
            text="Lead Count",
            title="Historical Lead Outcomes",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("About the Project")

    st.write(
        """
        The Lead Intent & Priority Scorer is a Machine Learning
        prototype that estimates the probability of a lead converting.

        The system learns patterns from historical lead records and
        uses those patterns to score new records. Counsellors can use
        the scores to help organize follow-up activities.
        """
    )

    st.warning(
        "Priority thresholds are illustrative. "
        "Model performance and real-world usefulness depend on "
        "the dataset, feature quality, and validation."
    )


# -------------------------------------------------
# 8. LEAD ANALYTICS
# -------------------------------------------------

elif page == "Lead Analytics":

    st.title("📊 Lead Analytics")
    st.write("Explore historical lead information.")

    converted_df = df.copy()

    categorical_columns = converted_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    categorical_columns = [
        column for column in categorical_columns
        if converted_df[column].nunique(dropna=True) <= 30
    ]

    if categorical_columns:

        selected_column = st.selectbox(
            "Select a category for analysis",
            categorical_columns,
        )

        chart_df = converted_df.copy()

        chart_df[selected_column] = (
            chart_df[selected_column]
            .fillna("Missing")
            .astype(str)
        )

        grouped = (
            chart_df.groupby(selected_column, dropna=False)
            .agg(
                Total_Leads=("Converted", "size"),
                Converted_Leads=("Converted", "sum"),
            )
            .reset_index()
        )

        grouped["Conversion Rate (%)"] = (
            grouped["Converted_Leads"]
            / grouped["Total_Leads"]
            * 100
        ).round(2)

        grouped = grouped.sort_values(
            "Total_Leads",
            ascending=False,
        )

        fig = px.bar(
            grouped,
            x=selected_column,
            y="Total_Leads",
            color="Conversion Rate (%)",
            hover_data=[
                "Converted_Leads",
                "Conversion Rate (%)",
            ],
            title=f"Leads by {selected_column}",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category Summary")
        st.dataframe(grouped, use_container_width=True)

    else:
        st.info(
            "No suitable low-cardinality categorical columns "
            "were found for this chart."
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )

    st.caption(
        "The analytics use historical outcomes. They do not "
        "represent predictions for future leads."
    )


# -------------------------------------------------
# 9. SCORE AN INDIVIDUAL LEAD
# -------------------------------------------------

elif page == "Score a Lead":

    st.title("🎯 Score an Individual Lead")

    st.write(
        "Enter the available lead information to estimate "
        "the probability of conversion."
    )

    st.info(
        "Only features used during training are included. "
        "Missing inputs are handled by the saved preprocessing pipeline."
    )

    record = {}

    with st.form("lead_scoring_form"):

        columns = st.columns(2)

        for index, feature in enumerate(feature_names):

            series = df[feature] if feature in df.columns else pd.Series(dtype="object")

            with columns[index % 2]:

                if pd.api.types.is_numeric_dtype(series):

                    values = pd.to_numeric(
                        series,
                        errors="coerce",
                    ).dropna()

                    default_value = (
                        float(values.median())
                        if not values.empty
                        else 0.0
                    )

                    record[feature] = st.number_input(
                        feature,
                        value=default_value,
                        key=f"input_{feature}",
                    )

                else:

                    options = (
                        series.dropna()
                        .astype(str)
                        .value_counts()
                        .head(30)
                        .index.tolist()
                    )

                    options = sorted(options)

                    options.insert(0, "Unknown / Not provided")

                    selected = st.selectbox(
                        feature,
                        options,
                        key=f"input_{feature}",
                    )

                    if selected == "Unknown / Not provided":
                        record[feature] = np.nan
                    else:
                        record[feature] = selected

        submitted = st.form_submit_button(
            "Calculate Lead Score",
            use_container_width=True,
        )

    if submitted:

        try:
            model_input = prepare_input(record)

            probability = float(
                model.predict_proba(model_input)[0, 1]
            )

            priority = get_priority(probability)

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            col1.metric(
                "Estimated Conversion Probability",
                f"{probability * 100:.2f}%",
            )

            col2.metric(
                "Assigned Priority",
                priority,
            )

            st.progress(float(np.clip(probability, 0, 1)))

            st.write(
                f"Estimated conversion probability: "
                f"**{probability * 100:.2f}%**"
            )

            if priority == "High":
                st.success(
                    "High priority: consider reviewing this lead "
                    "for timely follow-up."
                )
            elif priority == "Medium":
                st.warning(
                    "Medium priority: consider further engagement "
                    "and monitoring."
                )
            else:
                st.info(
                    "Low priority: consider additional information "
                    "or a suitable follow-up strategy."
                )

            st.caption(
                "This is a model estimate, not a guarantee of "
                "conversion or an assessment of a student's worth."
            )

        except Exception as error:
            st.error(f"Could not score this lead: {error}")


# -------------------------------------------------
# 10. BULK LEAD SCORING
# -------------------------------------------------

elif page == "Bulk Lead Scoring":

    st.title("📁 Bulk Lead Scoring")

    st.write(
        "Upload a CSV file containing multiple leads. "
        "The application will estimate a conversion probability "
        "for each row."
    )

    st.info(
        "The uploaded CSV should contain the feature columns "
        "used by the trained model. Missing features are filled "
        "with missing values and handled by the pipeline."
    )

    uploaded_file = st.file_uploader(
        "Upload leads CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:
            upload_df = pd.read_csv(uploaded_file)
            upload_df.columns = upload_df.columns.str.strip()

            st.subheader("Uploaded Data Preview")
            st.dataframe(
                upload_df.head(20),
                use_container_width=True,
            )

            st.write(f"Rows uploaded: {len(upload_df):,}")

            if st.button(
                "Score All Leads",
                use_container_width=True,
            ):

                if upload_df.empty:
                    st.warning("The uploaded CSV is empty.")

                else:
                    with st.spinner("Scoring uploaded leads..."):

                        scored_df = score_records(upload_df)

                    st.success("Lead scoring completed!")

                    high_count = int(
                        (scored_df["Priority"] == "High").sum()
                    )
                    medium_count = int(
                        (scored_df["Priority"] == "Medium").sum()
                    )
                    low_count = int(
                        (scored_df["Priority"] == "Low").sum()
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric("High Priority", high_count)
                    col2.metric("Medium Priority", medium_count)
                    col3.metric("Low Priority", low_count)

                    st.subheader("Scored Leads")

                    st.dataframe(
                        scored_df,
                        use_container_width=True,
                    )

                    csv_data = scored_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Scored Leads CSV",
                        data=csv_data,
                        file_name="scored_leads.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as error:
            st.error(
                f"Could not process the uploaded CSV: {error}"
            )



# -------------------------------------------------
# 11. MODEL PERFORMANCE
# -------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance")
    st.caption(
        "Evaluate the lead-scoring model using key machine learning metrics."
    )

    if metrics is None:
        st.warning(
            "Model metrics are unavailable. Run python train.py first."
        )
        st.stop()

    st.info(
        "The following metrics were calculated using the held-out test dataset."
    )

    # ---------------------------------------------
    # SELECTED MODEL
    # ---------------------------------------------

    st.subheader("🏆 Selected Model")

    st.metric(
        label="Best Performing Model",
        value=metrics.get("selected_model", "Unknown"),
    )

    st.divider()

    # ---------------------------------------------
    # MODEL EVALUATION METRICS
    # ---------------------------------------------

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Accuracy",
        value=f"{metrics.get('accuracy', 0) * 100:.2f}%",
        help="Overall proportion of correct predictions."
    )

    col2.metric(
        label="Precision",
        value=f"{metrics.get('precision', 0) * 100:.2f}%",
        help="How many predicted converted leads actually converted."
    )

    col3.metric(
        label="Recall",
        value=f"{metrics.get('recall', 0) * 100:.2f}%",
        help="How many actual converted leads the model identified."
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        label="F1 Score",
        value=f"{metrics.get('f1_score', 0) * 100:.2f}%",
        help="Balance between precision and recall."
    )

    col5.metric(
        label="ROC-AUC",
        value=f"{metrics.get('roc_auc', 0) * 100:.2f}%",
        help="Measures how well the model distinguishes between classes."
    )

    col6.metric(
        label="Input Features",
        value=metrics.get("feature_count", 0),
        help="Number of features used by the model."
    )

    st.divider()

    # ---------------------------------------------
    # MODEL COMPARISON
    # ---------------------------------------------

    st.subheader("🔍 Model Comparison")

    comparison_path = MODEL_PATH.parent / "model_comparison.csv"

    if comparison_path.exists():

        comparison_df = pd.read_csv(comparison_path)

        # Format model comparison table
        format_columns = {
            col: "{:.3f}"
            for col in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC AUC",
            ]
            if col in comparison_df.columns
        }

        st.dataframe(
            comparison_df.style.format(format_columns),
            use_container_width=True,
            hide_index=True,
        )

        required_columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
        ]

        if all(
            column in comparison_df.columns
            for column in required_columns
        ):

            chart_df = comparison_df.melt(
                id_vars="Model",
                value_vars=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC AUC",
                ],
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                chart_df,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Performance Comparison Across Models",
                range_y=[0, 1],
                text_auto=".2f",
            )

            fig.update_layout(
                xaxis_title="Evaluation Metric",
                yaxis_title="Score",
                legend_title="Machine Learning Model",
                template="plotly_white",
                height=450,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.warning(
                "The model comparison file is missing required columns. "
                "Check models/model_comparison.csv."
            )

    else:
        st.warning(
            "Model comparison data was not found. "
            "Check whether models/model_comparison.csv exists."
        )

    st.divider()

    # ---------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------

    st.subheader("🎯 Confusion Matrix")

    st.caption(
        "Compare actual outcomes with the model's predicted outcomes."
    )

    matrix = metrics.get("confusion_matrix")

    if (
        matrix is not None
        and len(matrix) == 2
        and all(len(row) == 2 for row in matrix)
    ):

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                "Actual: Not Converted",
                "Actual: Converted",
            ],
            columns=[
                "Predicted: Not Converted",
                "Predicted: Converted",
            ],
        )

        st.dataframe(
            matrix_df.style.format("{:,.0f}"),
            use_container_width=True,
        )

    else:
        st.info(
            "Confusion matrix is unavailable. "
            "Run python train.py to generate model metrics."
        )

    st.divider()

    # ---------------------------------------------
    # TRAINING INFORMATION
    # ---------------------------------------------

    
    # ---------------------------------------------
    # TRAINING INFORMATION
    # ---------------------------------------------

    st.subheader("📋 Training Information")

    st.caption(
        "Overview of the dataset used to train and evaluate the model."
    )

    train_records = metrics.get("train_records")
    test_records = metrics.get("test_records")
    feature_count = metrics.get("feature_count")
    conversion_rate = metrics.get("conversion_rate")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Training Records",
            value=(
                f"{int(train_records):,}"
                if train_records is not None
                else "N/A"
            ),
        )

    with col2:
        st.metric(
            label="Testing Records",
            value=(
                f"{int(test_records):,}"
                if test_records is not None
                else "N/A"
            ),
        )

    with col3:
        st.metric(
            label="Feature Count",
            value=(
                feature_count
                if feature_count is not None
                else "N/A"
            ),
        )

    with col4:
        st.metric(
            label="Historical Conversion",
            value=(
                format_percentage(conversion_rate)
                if conversion_rate is not None
                else "N/A"
            ),
        )


    # ---------------------------------------------
    # MODEL PERFORMANCE NOTE
    # ---------------------------------------------

    st.caption(
        "Note: These results reflect performance on held-out test data. "
        "Actual performance may vary when scoring new leads."
    )




# -------------------------------------------------
# 12. FOOTER
# -------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("Lead Intent & Priority Scorer")
st.sidebar.caption("Educational Machine Learning prototype")