
# dashboard_utils.py
import streamlit as st
from PIL import Image
import requests
from src.training.train_pipeline import TrainingPipeline
from src.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH

def train_pipeline(name, serialize):
    with st.spinner('Training pipeline, please wait...'):
        try:
            tp = TrainingPipeline()
            tp.train(serialize=serialize, model_name=name)
            tp.render_confusion_matrix()
            accuracy, f1 = tp.get_model_performance()
            display_metrics(accuracy, f1)
        except Exception as e:
            st.error('Failed to train the pipeline!')
            st.exception(e)

def display_metrics(accuracy, f1):
    col1, col2 = st.columns(2)
    col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
    col2.metric(label="F1 score", value=str(round(f1, 4)))
    st.image(Image.open(CM_PLOT_PATH), width=850)

def run_inference(sample):
    with st.spinner('Running inference...'):
        try:
            sample_file = "_".join(sample.upper().split()) + ".txt"
            with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                sample_text = file.read()

            result = requests.post(
                'http://localhost:9000/api/inference',
                json={'text': sample_text}
            )
            st.success('Done!')
            label = LABELS_MAP.get(int(float(result.text)))
            st.metric(label="Status", value=f"Resume label: {label}")
        except Exception as e:
            st.error('Failed to call Inference API!')
            st.exception(e)





