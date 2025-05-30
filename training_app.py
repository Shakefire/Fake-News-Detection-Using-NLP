from visualization import create_confidence_meter, create_bias_chart, create_sentiment_radar
from prediction_system import PredictionSystem
from model_trainer import FakeNewsTrainer
from data_loader import FakeNewsNetLoader, DataValidator
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="FakeNewsNet Model Training",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state for all relevant variables
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'test_text' not in st.session_state:
    st.session_state.test_text = ""
if 'news_text' not in st.session_state:
    st.session_state.news_text = ""

# Main title
st.title("🤖 NewsAnalyzer: Model Training & Prediction System")
st.markdown(
    "NewsAnalyzer - Train models on FakeNewsNet dataset and make predictions on new articles")

# Sidebar navigation
st.sidebar.title("NewsAnalyzer Training Pipeline")
page = st.sidebar.selectbox("Choose a step:", [
    "1. Data Loading",
    "2. Model Training",
    "3. Model Testing",
    "4. Prediction System",
    "5. Model Management"
])

# Initialize components in session state to persist across reruns
if 'loader' not in st.session_state:
    st.session_state.loader = FakeNewsNetLoader()
if 'trainer' not in st.session_state:
    st.session_state.trainer = FakeNewsTrainer()
if 'predictor' not in st.session_state:
    st.session_state.predictor = PredictionSystem()

loader = st.session_state.loader
trainer = st.session_state.trainer
predictor = st.session_state.predictor

if page == "1. Data Loading":
    st.header("📊 NewsAnalyzer Data Loading")

    st.info("This will load the FakeNewsNet dataset from the official repository. The dataset contains news articles labeled as fake or real from GossipCop and PolitiFact.")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Load FakeNewsNet Dataset", type="primary"):
            with st.spinner("Loading FakeNewsNet dataset..."):
                # Load the datasets
                datasets = loader.load_or_download_data()

                if datasets:
                    st.success("Dataset loaded successfully!")

                    # Show dataset overview
                    st.subheader("Dataset Overview")
                    for name, df in datasets.items():
                        if not df.empty:
                            st.write(
                                f"**{name.replace('_', ' ').title()}**: {len(df)} articles")

                    # Prepare training data
                    with st.spinner("Preparing training data..."):
                        training_data = loader.prepare_training_data(datasets)

                        if not training_data.empty:
                            # Clean the data
                            training_data = DataValidator.clean_training_data(
                                training_data)
                            st.session_state.training_data = training_data

                            st.success(
                                f"Training data prepared: {len(training_data)} samples")

                            # Show data validation results
                            validation_results = DataValidator.validate_training_data(
                                training_data)

                            st.subheader("Data Quality Report")

                            val_col1, val_col2, val_col3 = st.columns(3)
                            with val_col1:
                                st.metric("Total Samples",
                                          validation_results['total_samples'])
                                st.metric(
                                    "Fake News", validation_results['fake_samples'])
                            with val_col2:
                                st.metric(
                                    "Real News", validation_results['real_samples'])
                                st.metric(
                                    "Avg Text Length", f"{validation_results['avg_text_length']:.0f}")
                            with val_col3:
                                st.metric("Duplicates Removed",
                                          validation_results['duplicate_texts'])
                                st.metric("Empty Texts Removed",
                                          validation_results['empty_texts'])

                            # Show sample data
                            st.subheader("Sample Data")
                            sample_data = training_data.sample(
                                min(10, len(training_data)))
                            st.dataframe(
                                sample_data[['text', 'label', 'source']])
                        else:
                            st.error(
                                "No valid training data could be prepared")
                else:
                    st.error("Failed to load dataset")

    with col2:
        st.subheader("Dataset Information")
        st.info("""
        **FakeNewsNet** is a repository for fake news detection research:
        
        - **GossipCop**: Entertainment news
        - **PolitiFact**: Political news
        - Each contains both fake and real articles
        - Articles are labeled by fact-checkers
        
        The dataset will be downloaded from the official GitHub repository.
        """)

elif page == "2. Model Training":
    st.header("🎯 NewsAnalyzer Model Training")

    if st.session_state.training_data is None:
        st.warning(
            "Please load the training data first from the Data Loading page.")
        st.stop()

    st.success(
        f"Training data ready: {len(st.session_state.training_data)} samples")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Training Configuration")

        # Model selection
        model_type = st.selectbox(
            "Select Model Type:",
            ["logistic", "naive_bayes", "random_forest"],
            help="Choose the machine learning algorithm for training"
        )

        # Training parameters
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

        # Model name
        model_name = st.text_input(
            "Model Name:",
            value=f"fakenews_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Name for saving the trained model"
        )

        # Training button
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Train the model
                    training_info = trainer.train_model(
                        st.session_state.training_data,
                        model_type=model_type,
                        test_size=test_size
                    )

                    if training_info:
                        st.session_state.model_trained = True

                        # Save the model
                        if trainer.save_model(model_name):
                            st.session_state.current_model = model_name
                            st.success(
                                f"Model '{model_name}' trained and saved successfully!")

                            # Show training results
                            st.subheader("Training Results")

                            result_col1, result_col2, result_col3 = st.columns(
                                3)
                            with result_col1:
                                st.metric(
                                    "Training Accuracy", f"{training_info['train_accuracy']:.4f}")
                            with result_col2:
                                st.metric(
                                    "Test Accuracy", f"{training_info['test_accuracy']:.4f}")
                            with result_col3:
                                st.metric(
                                    "Features", training_info['feature_count'])
                        else:
                            st.error(
                                "Model training completed but saving failed")
                    else:
                        st.error("Model training failed")

                except Exception as e:
                    st.error(f"Training error: {str(e)}")

    with col2:
        st.subheader("Model Information")

        if model_type == "logistic":
            st.info("""
            **NewsAnalyzer Logistic Regression**
            - Fast training and prediction
            - Good interpretability
            - Works well with text features
            - Provides feature importance
            """)
        elif model_type == "naive_bayes":
            st.info("""
            **NewsAnalyzer Naive Bayes**
            - Very fast training
            - Good for text classification
            - Handles large vocabularies well
            - Assumes feature independence
            """)
        elif model_type == "random_forest":
            st.info("""
            **NewsAnalyzer Random Forest**
            - Robust to overfitting
            - Handles feature interactions
            - Good performance on many tasks
            - Less interpretable
            """)

elif page == "3. Model Testing":
    st.header("🧪 NewsAnalyzer Model Testing")

    # Load model selection
    available_models = trainer.list_saved_models()

    if not available_models:
        st.warning("No trained models found. Please train a model first.")
        st.stop()

    selected_model = st.selectbox("Select Model to Test:", available_models)

    if st.button("Load Model"):
        if trainer.load_model(selected_model):
            st.session_state.current_model = selected_model

            # Show model info
            model_info = trainer.get_model_info()
            st.subheader("Model Information")

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                for key, value in list(model_info.items())[:3]:
                    st.metric(key, value)
            with info_col2:
                for key, value in list(model_info.items())[3:]:
                    st.metric(key, value)

    # Test with custom text
    if st.session_state.current_model:
        st.subheader("Test with Custom News")

        test_text = st.text_area(
            "Enter news text to test:",
            height=150,
            placeholder="Paste a news article here to test the model..."
        )

        if st.button("Predict", disabled=not test_text):
            with st.spinner("Making prediction..."):
                prediction, confidence = trainer.predict(test_text)

                if prediction:
                    # Display prediction
                    pred_col1, pred_col2 = st.columns(2)

                    with pred_col1:
                        if prediction == "Real":
                            st.success(f"**Prediction: {prediction}**")
                        else:
                            st.error(f"**Prediction: {prediction}**")

                    with pred_col2:
                        # Confidence meter
                        fig_confidence = create_confidence_meter(confidence)
                        st.plotly_chart(
                            fig_confidence, use_container_width=True)

                    # Feature importance
                    feature_importance = trainer.get_feature_importance(
                        test_text)
                    if feature_importance:
                        st.subheader("Important Words")

                        # Create bar chart
                        words = list(feature_importance.keys())[:10]
                        importances = list(feature_importance.values())[:10]
                        colors = ['red' if imp >
                                  0 else 'blue' for imp in importances]

                        fig = go.Figure(data=[
                            go.Bar(
                                x=[abs(imp) for imp in importances],
                                y=words,
                                orientation='h',
                                marker_color=colors
                            )
                        ])
                        fig.update_layout(
                            title="Feature Importance (Red=Fake indicators, Blue=Real indicators)",
                            height=400,
                            margin=dict(l=120, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

elif page == "4. Prediction System":
    st.header("🔮 NewsAnalyzer News Prediction System")

    # Model selection
    available_models = trainer.list_saved_models()

    if not available_models:
        st.warning("No trained models found. Please train a model first.")
        st.stop()

    # Load model
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_model = st.selectbox("Select Model:", available_models)

        if st.button("Load Model for Predictions"):
            if predictor.load_model(selected_model):
                st.session_state.current_model = selected_model

    if st.session_state.current_model:
        st.success(
            f"Model '{st.session_state.current_model}' loaded for predictions")

        # Prediction interface
        st.subheader("Make Predictions")

        # Single prediction
        with st.expander("Single Article Prediction", expanded=True):
            news_text = st.text_area(
                "Enter news article:",
                height=200,
                placeholder="Paste the news article text here..."
            )

            if st.button("Analyze Article", disabled=not news_text):
                with st.spinner("Analyzing article..."):
                    result = predictor.predict_news(news_text)

                    if result:
                        # Display comprehensive results
                        st.subheader("Analysis Results")

                        # Main prediction
                        pred_col1, pred_col2 = st.columns(2)

                        with pred_col1:
                            if result['prediction'] == "Real":
                                st.success(
                                    f"**Classification: {result['prediction']}**")
                            else:
                                st.error(
                                    f"**Classification: {result['prediction']}**")

                            st.metric("Confidence",
                                      f"{result['confidence']:.1%}")

                        with pred_col2:
                            # Confidence visualization
                            fig_conf = create_confidence_meter(
                                result['confidence'])
                            st.plotly_chart(fig_conf, use_container_width=True)

                        # Additional analysis
                        analysis_col1, analysis_col2 = st.columns(2)

                        with analysis_col1:
                            st.subheader("Sentiment Analysis")
                            sentiment = result['sentiment']
                            st.write(
                                f"**Polarity:** {sentiment['polarity']:.2f}")
                            st.write(
                                f"**Subjectivity:** {sentiment['subjectivity']:.2f}")
                            st.write(f"**Label:** {sentiment['label']}")

                        with analysis_col2:
                            st.subheader("Bias Analysis")
                            bias = result['bias']
                            st.write(
                                f"**Political Bias:** {bias['political_bias']:.2f}")
                            st.write(
                                f"**Emotional Intensity:** {bias['emotional_intensity']:.2f}")
                            st.write(f"**Tone:** {bias['tone']}")

        # Bulk prediction
        with st.expander("Bulk Prediction"):
            uploaded_file = st.file_uploader(
                "Upload CSV file with news articles",
                type=['csv'],
                help="CSV should have a 'text' column with news articles"
            )

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(df)} articles")

                    if 'text' in df.columns:
                        if st.button("Run Bulk Prediction"):
                            progress_bar = st.progress(0)

                            results = predictor.bulk_predict(
                                df['text'].tolist(),
                                progress_bar=progress_bar
                            )

                            st.success(
                                f"Completed predictions for {len(results)} articles")

                            # Show results summary
                            fake_count = sum(
                                1 for r in results if r['prediction'] == 'Fake')
                            real_count = len(results) - fake_count

                            summary_col1, summary_col2, summary_col3 = st.columns(
                                3)
                            with summary_col1:
                                st.metric("Total Articles", len(results))
                            with summary_col2:
                                st.metric("Fake News", fake_count)
                            with summary_col3:
                                st.metric("Real News", real_count)
                    else:
                        st.error("CSV file must contain a 'text' column")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

    with col2:
        # Prediction statistics
        st.subheader("Prediction Statistics")
        stats = predictor.get_prediction_stats()

        st.metric("Total Predictions", stats['total_predictions'])
        if stats['total_predictions'] > 0:
            st.metric("Fake News %", f"{stats['fake_percentage']:.1f}%")
            st.metric("Real News %", f"{stats['real_percentage']:.1f}%")
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")

elif page == "5. Model Management":
    st.header("⚙️ NewsAnalyzer Model Management")

    # List saved models
    available_models = trainer.list_saved_models()

    if available_models:
        st.subheader("Saved Models")

        for model_name in available_models:
            with st.expander(f"Model: {model_name}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button(f"Load {model_name}", key=f"load_{model_name}"):
                        if trainer.load_model(model_name):
                            model_info = trainer.get_model_info()
                            st.json(model_info)

                with col2:
                    if st.button(f"Test {model_name}", key=f"test_{model_name}"):
                        st.info("Go to Model Testing page to test this model")

                with col3:
                    if st.button(f"Delete {model_name}", key=f"delete_{model_name}", type="secondary"):
                        st.warning(
                            "Delete functionality would be implemented here")
    else:
        st.info("No saved models found. Train a model first.")

    # Export predictions
    st.subheader("Export NewsAnalyzer Predictions")

    stats = predictor.get_prediction_stats()

    if stats['total_predictions'] > 0:
        export_format = st.selectbox("Export Format:", ["csv", "json"])

        if st.button("Export Predictions"):
            export_file = predictor.export_predictions(export_format)
            if export_file:
                st.success(f"Predictions exported to: {export_file}")
            else:
                st.error("Export failed")

        # Clear history
        if st.button("Clear Prediction History", type="secondary"):
            predictor.clear_history()
    else:
        st.info("No predictions to export")
