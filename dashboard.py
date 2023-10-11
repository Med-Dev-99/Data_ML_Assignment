
import time
import streamlit as st
from dashboard_utils import train_pipeline, run_inference
from src.constants import LABELS_MAP
import matplotlib.pyplot as plt
import seaborn as sns
from src.training.custom_training import CustomTrainingPipeline  # Import custom training pipeline
import pandas as pd
from src.constants import RAW_DATASET_PATH  # Import dataset path
import sqlite3
from src.constants import SAMPLES_PATH


# Streamlit app setup
st.title("Resume Classification Dashboard")
mode = st.sidebar.selectbox("Dashboard Modes", ("EDA", "Training", "Inference"))


# Create an SQLite database and cursor
conn = sqlite3.connect('prediction_results.db')
cursor = conn.cursor()

# Check if the table exists, and create it if not
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        resume_text TEXT,
        predicted_label TEXT
    )
''')
conn.commit()

# Function to run inference and update the database
def run_inference(sample_text):
    try:
        # Perform inference with your custom pipeline or baseline pipeline
        result = custom_pipeline.predict(sample_text)  # Replace with your inference method

        # Save the prediction result in the SQLite database
        cursor.execute('INSERT INTO predictions (resume_text, predicted_label) VALUES (?, ?)', (sample_text, result))
        conn.commit()
        return result
    except Exception as e:
        st.error('Failed to call Inference API!')
        st.exception(e)
        return None



if mode == "EDA":
    df = pd.read_csv(RAW_DATASET_PATH)

    # Streamlit Title
    st.title("Resume Classification Dashboard")
    st.header("Exploratory Data Analysis")
    st.info("In this section, you are invited to create insightful graphs about the resume dataset that you were provided.")
    # Create and display charts
    st.subheader("Data Visualization")

    # Count of each label
    st.write("Label Distribution:")
    label_counts = df['label'].value_counts()
    st.bar_chart(label_counts)

    # Distribution of resume lengths
    df['resume_length'] = df['resume'].apply(len)
    st.write("Resume Length Distribution:")
    st.hist(df['resume_length'], bins=20, color='skyblue', edgecolor='black')
    st.pyplot()

    # Pairwise correlation heatmap
    st.write("Correlation Heatmap:")
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot()
elif mode == "Training":
    st.header("Custom Pipeline Training")
    st.info("This section allows you to train a custom pipeline and evaluate its performance.")

    data = pd.read_csv(RAW_DATASET_PATH)  # Load your dataset
    custom_pipeline = CustomTrainingPipeline(data)

    name = st.text_input('Pipeline name', value='custom_pipeline')
    serialize = st.checkbox('Save pipeline')
    train = st.button('Train pipeline')

    if train:
        custom_pipeline.train(serialize=serialize, model_name=name)

        # Display performance metrics
        accuracy, f1 = custom_pipeline.get_model_performance()
        custom_pipeline.render_confusion_matrix()
        st.write(f'Accuracy: {accuracy:.4f}')
        st.write(f'F1 Score: {f1:.4f}')

elif mode == "Inference":
    st.header("Resume Inference")
    st.info("This section simplifies the inference process. Choose a test resume and observe the label that your trained pipeline will predict.")

    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    infer = st.button('Run Inference')

    if infer:
        with st.spinner('Running inference...'):
            sample_file = "_".join(sample.upper().split()) + ".txt"
            with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                sample_text = file.read()

            result = run_inference(sample_text)

            if result is not None:
                st.success('Inference completed!')
                label = LABELS_MAP.get(int(float(result)))
                st.metric(label="Status", value=f"Resume label: {label}")

                # Display the contents of the SQLite table
                cursor.execute('SELECT * FROM predictions')
                result_table = cursor.fetchall()
                st.subheader("Prediction Results")
                st.dataframe(pd.DataFrame(result_table, columns=['ID', 'Resume Text', 'Predicted Label']), height=300)

else:
    st.header("Resume Inference")
    st.info("This section simplifies the inference process. Choose a test resume and observe the label that your trained pipeline will predict.")

    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    infer = st.button('Run Inference')

    if infer:
        run_inference(sample)

if __name__ == '__main__':
    main()
