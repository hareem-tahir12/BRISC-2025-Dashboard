
import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="BRISC 2025 | Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #F7F9FC;
    color: #17233C;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E4E8F0;
}

section[data-testid="stSidebar"] * {
    color: #17233C !important;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #17233C;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #64748B;
    margin-bottom: 30px;
}

.card {
    background: #FFFFFF;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #E4E8F0;
    margin-bottom: 20px;
}

.metric-card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E4E8F0;
    text-align: center;
}

.metric-title {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 8px;
}

.metric-value {
    color: #315A8A;
    font-size: 28px;
    font-weight: 700;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #17233C;
    margin-top: 20px;
    margin-bottom: 15px;
}

.workflow-box {
    background: #FFFFFF;
    border: 1px solid #DCE3EE;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
}

.workflow-title {
    font-size: 19px;
    font-weight: 700;
    color: #315A8A;
    margin-bottom: 12px;
}

.workflow-step {
    background: #F7F9FC;
    border: 1px solid #E2E8F0;
    padding: 12px;
    border-radius: 8px;
    margin: 7px 0;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #64748B;
    padding: 30px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# DATA
# =========================

dataset = pd.DataFrame({
    "Class": [
        "Glioma",
        "Meningioma",
        "No Tumor",
        "Pituitary"
    ],
    "Images": [
        1147,
        1329,
        1067,
        1457
    ]
})

model_results = pd.DataFrame({
    "Model": [
        "KNN",
        "Random Forest",
        "Logistic Regression",
        "Decision Tree",
        "EfficientNetB7"
    ],
    "Accuracy": [
        89.10,
        88.70,
        90.90,
        66.70,
        82.20
    ],
    "Macro Precision": [
        89.95,
        89.88,
        91.18,
        68.70,
        83.26
    ],
    "Macro Recall": [
        90.40,
        89.74,
        91.82,
        68.00,
        84.38
    ],
    "Macro F1": [
        89.86,
        89.65,
        91.35,
        67.14,
        83.15
    ]
})


# =========================
# SIDEBAR
# =========================

st.sidebar.markdown(
    "<h2>🧠 BRISC 2025</h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "Brain Tumor Classification<br>Research Dashboard",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Dataset",
        "Model Performance",
        "EfficientNetB7",
        "Methodology",
        "Project Information"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("BRISC 2025 • ML Research Project")


# =========================
# OVERVIEW
# =========================

if page == "Overview":

    st.markdown(
        '<div class="main-title">BRISC 2025 Brain Tumor Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'A research dashboard for comparative brain MRI classification using '
        'traditional machine learning and deep learning.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Tumor Classes</div>
        <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Training Images</div>
        <div class="metric-value">5,000</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">ML Models</div>
        <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Deep Model</div>
        <div class="metric-value">B7</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
    <p>
    The BRISC 2025 project investigates automated brain tumor classification
    from MRI images. The study compares four traditional machine learning
    classifiers with an EfficientNetB7 deep learning approach.
    </p>

    <p>
    The classification task contains four categories:
    <b>Glioma, Meningioma, No Tumor, and Pituitary.</b>
    </p>

    <p>
    The traditional pipeline uses handcrafted HOG features followed by PCA,
    while the deep learning pipeline uses an ImageNet-pretrained EfficientNetB7
    architecture.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        model_results,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Classification Accuracy Comparison"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Accuracy (%)",
        xaxis_title="Model",
        yaxis_range=[0, 100],
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# DATASET
# =========================

elif page == "Dataset":

    st.markdown(
        '<div class="main-title">Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">BRISC 2025 training dataset distribution</div>',
        unsafe_allow_html=True
    )

    total_images = dataset["Images"].sum()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Total Training Images</div>
        <div class="metric-value">5,000</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Number of Classes</div>
        <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Training Image Distribution</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        dataset,
        x="Class",
        y="Images",
        text="Images",
        title="BRISC Dataset Image Distribution"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Tumor Class",
        yaxis_title="Number of Images",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Class Distribution</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        dataset,
        use_container_width=True,
        hide_index=True
    )


# =========================
# MODEL PERFORMANCE
# =========================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Comparative performance of the evaluated classification models'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Performance Summary</div>',
        unsafe_allow_html=True
    )

    display_df = model_results.copy()

    display_df["Accuracy"] = display_df["Accuracy"].map(lambda x: f"{x:.2f}%")
    display_df["Macro Precision"] = display_df["Macro Precision"].map(lambda x: f"{x:.2f}%")
    display_df["Macro Recall"] = display_df["Macro Recall"].map(lambda x: f"{x:.2f}%")
    display_df["Macro F1"] = display_df["Macro F1"].map(lambda x: f"{x:.2f}%")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Accuracy Comparison</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        model_results,
        x="Model",
        y="Accuracy",
        text="Accuracy"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Macro Metrics</div>',
        unsafe_allow_html=True
    )

    metrics_df = model_results.melt(
        id_vars="Model",
        value_vars=[
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],
        var_name="Metric",
        value_name="Score"
    )

    fig2 = px.bar(
        metrics_df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group"
    )

    fig2.update_layout(
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)


# =========================
# EFFICIENTNET
# =========================

elif page == "EfficientNetB7":

    st.markdown(
        '<div class="main-title">EfficientNetB7</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Deep learning approach for four-class brain MRI classification'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Architecture</div>
        <div class="metric-value">B7</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Input Size</div>
        <div class="metric-value">600×600</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Dropout</div>
        <div class="metric-value">0.3</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">Output Classes</div>
        <div class="metric-value">4</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
    <p>
    EfficientNetB7 uses <b>ImageNet pretrained weights</b> as the feature
    extraction backbone.
    </p>

    <p>
    The architecture processes 600×600 RGB MRI images, followed by Global
    Average Pooling, Dropout (0.3), and a Dense Softmax layer for four-class
    classification.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Reported Performance</div>',
        unsafe_allow_html=True
    )

    b7_metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ],
        "Score": [
            "82.20%",
            "83.26%",
            "84.38%",
            "83.15%"
        ]
    })

    st.dataframe(
        b7_metrics,
        use_container_width=True,
        hide_index=True
    )


# =========================
# METHODOLOGY
# =========================

elif page == "Methodology":

    st.markdown(
        '<div class="main-title">Methodology</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Research workflow used for traditional ML and deep learning'
        '</div>',
        unsafe_allow_html=True
    )

    # Traditional ML
    st.markdown("""
    <div class="workflow-box">

    <div class="workflow-title">
    A. Traditional Machine Learning Pipeline
    </div>

    <div class="workflow-step">
    <b>1.</b> MRI Images
    </div>

    <div class="workflow-step">
    <b>2.</b> Grayscale Conversion
    </div>

    <div class="workflow-step">
    <b>3.</b> Resize (128×128)
    </div>

    <div class="workflow-step">
    <b>4.</b> Normalization [0,1]
    </div>

    <div class="workflow-step">
    <b>5.</b> HOG Feature Extraction (8100 features)
    </div>

    <div class="workflow-step">
    <b>6.</b> PCA (95% variance)
    </div>

    <div class="workflow-step">
    <b>7.</b> ML Classifiers
    </div>

    <div class="workflow-step">
    KNN • Random Forest • Logistic Regression • Decision Tree
    </div>

    <div class="workflow-step">
    <b>8.</b> Prediction
    </div>

    </div>
    """, unsafe_allow_html=True)

    # Deep Learning
    st.markdown("""
    <div class="workflow-box">

    <div class="workflow-title">
    B. Deep Learning Pipeline
    </div>

    <div class="workflow-step">
    <b>1.</b> MRI Images
    </div>

    <div class="workflow-step">
    <b>2.</b> Resize & Normalize (600×600 RGB)
    </div>

    <div class="workflow-step">
    <b>3.</b> EfficientNetB7 (ImageNet pretrained weights)
    </div>

    <div class="workflow-step">
    <b>4.</b> Global Average Pooling
    </div>

    <div class="workflow-step">
    <b>5.</b> Dropout (0.3)
    </div>

    <div class="workflow-step">
    <b>6.</b> Dense Softmax (4 classes)
    </div>

    <div class="workflow-step">
    <b>7.</b> Prediction
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Research Approach</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <p>
    The methodology evaluates two different approaches to brain tumor
    classification.
    </p>

    <p>
    The first approach extracts handcrafted HOG features from preprocessed
    MRI images and reduces their dimensionality using PCA before classification.
    </p>

    <p>
    The second approach uses transfer learning with EfficientNetB7 and
    ImageNet pretrained weights to learn image representations directly.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================
# PROJECT INFORMATION
# =========================

elif page == "Project Information":

    st.markdown(
        '<div class="main-title">Project Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">BRISC 2025 research project details</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>Project Title</h3>

    <p>
    <b>Brain Tumor Classification Using Machine Learning and Deep Learning</b>
    </p>

    <h3>Dataset</h3>

    <p>
    BRISC 2025 Brain MRI Dataset
    </p>

    <h3>Classification Classes</h3>

    <p>
    Glioma • Meningioma • No Tumor • Pituitary
    </p>

    <h3>Traditional Machine Learning</h3>

    <p>
    KNN, Random Forest, Logistic Regression, and Decision Tree.
    </p>

    <h3>Deep Learning</h3>

    <p>
    EfficientNetB7 with ImageNet pretrained weights.
    </p>

    <h3>Project Focus</h3>

    <p>
    Comparative evaluation of handcrafted-feature-based machine learning
    and transfer-learning-based deep learning approaches for brain MRI
    classification.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
    BRISC 2025 • Brain Tumor Classification Research Dashboard
    </div>
    """, unsafe_allow_html=True)
