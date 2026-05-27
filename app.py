import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from train_model import train_all_models


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()

    return fig


@st.cache_resource
def training(test_size, random_state):
    return train_all_models(test_size=test_size, random_state=random_state)


st.set_page_config(
    page_title="NB, GDA, LDA, QDA Classifier",
    page_icon="🍷",
    layout="wide"
)


st.title("Naive Bayes, GDA, LDA and QDA From Scratch")
st.markdown(
    """
This project implements and compares four probabilistic classification approaches:

- **Gaussian Naive Bayes from scratch**
- **GDA / LDA from scratch**
- **QDA from scratch**
- **Sklearn baseline models**

The dataset used is the **Wine dataset** from sklearn.
"""
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Experiment Settings")

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=999,
    value=42,
    step=1
)

show_dataset = st.sidebar.checkbox("Show Dataset", value=False)


# ============================================================
# Train Models
# ============================================================

output = training(test_size, random_state)

data = output["data"]
df = output["df"]
scaler = output["scaler"]
results_df = output["results_df"]
trained_models = output["trained_models"]


# ============================================================
# Dataset Overview
# ============================================================

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Samples", data.data.shape[0])

with col2:
    st.metric("Features", data.data.shape[1])

with col3:
    st.metric("Classes", len(data.target_names))

st.write("Class Names:", list(data.target_names))

if show_dataset:
    st.subheader("Wine Dataset")
    st.dataframe(df, use_container_width=True)


# ============================================================
# Accuracy Comparison
# ============================================================

st.header("2. Model Accuracy Comparison")

st.dataframe(results_df, use_container_width=True)

best_model = results_df.iloc[0]["Model"]
best_accuracy = results_df.iloc[0]["Accuracy"]

st.success(f"Best Model: {best_model} with accuracy {best_accuracy:.4f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(results_df["Model"], results_df["Accuracy"])
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
fig.tight_layout()

st.pyplot(fig)


# ============================================================
# Detailed Evaluation
# ============================================================

st.header("3. Detailed Model Evaluation")

selected_model_name = st.selectbox(
    "Select a model",
    list(trained_models.keys())
)

selected_model_data = trained_models[selected_model_name]

y_test = selected_model_data["y_test"]
y_pred = selected_model_data["y_pred"]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=data.target_names,
    output_dict=True
)

st.subheader(selected_model_name)
st.metric("Accuracy", f"{accuracy:.4f}")

col1, col2 = st.columns(2)

with col1:
    st.write("Confusion Matrix")
    st.pyplot(plot_confusion_matrix(cm, data.target_names))

with col2:
    st.write("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)


# ============================================================
# Manual Prediction
# ============================================================

st.header("4. Manual Wine Class Prediction")

st.markdown(
    """
Enter feature values below and select a model to predict the wine class.
"""
)

manual_model_name = st.selectbox(
    "Choose model for prediction",
    list(trained_models.keys()),
    key="manual_model"
)

input_values = []

with st.expander("Enter Feature Values", expanded=True):
    for feature_name, feature_values in zip(data.feature_names, data.data.T):
        value = st.slider(
            label=feature_name,
            min_value=float(np.min(feature_values)),
            max_value=float(np.max(feature_values)),
            value=float(np.mean(feature_values))
        )

        input_values.append(value)

if st.button("Predict Wine Class"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    selected_model = trained_models[manual_model_name]["model"]
    prediction = selected_model.predict(input_scaled)[0]

    predicted_class = data.target_names[prediction]

    st.success(f"Predicted Wine Class: {predicted_class}")
