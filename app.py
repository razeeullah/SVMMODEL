"""
Streamlit Web Application for SVM Iris Classification
Author: ML Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-result {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4A90E2;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and metadata"""
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, scaler, metadata
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run 'python train_model.py' first.")
        st.stop()

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🌸 Iris Flower Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Support Vector Machine (SVM)</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, metadata = load_model_artifacts()
    
    # Display model info
    with st.expander("ℹ️ About This Model"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{metadata['test_accuracy']*100:.2f}%")
        with col2:
            st.metric("Train Accuracy", f"{metadata['train_accuracy']*100:.2f}%")
        with col3:
            st.metric("CV Mean Score", f"{metadata['cv_mean_score']*100:.2f}%")
        
        st.info("""
        This model classifies Iris flowers into three species:
        - 🌺 **Setosa**
        - 🌻 **Versicolor**
        - 🌷 **Virginica**
        
        The model uses 4 features: sepal length, sepal width, petal length, and petal width.
        """)
    
    # Sidebar for inputs
    st.sidebar.header("🔧 Input Features")
    st.sidebar.markdown("Adjust the sliders to set flower measurements:")
    
    # Feature inputs
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.8,
        step=0.1,
        help="Length of the sepal in centimeters"
    )
    
    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.0,
        step=0.1,
        help="Width of the sepal in centimeters"
    )
    
    petal_length = st.sidebar.slider(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=3.8,
        step=0.1,
        help="Length of the petal in centimeters"
    )
    
    petal_width = st.sidebar.slider(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.2,
        step=0.1,
        help="Width of the petal in centimeters"
    )
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Input Summary")
        st.dataframe(input_data, use_container_width=True)
        
        # Predict button
        predict_button = st.button("🔮 Predict Species", use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Result")
        
        if predict_button:
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Get species name
            species = metadata['target_names'][prediction]
            confidence = prediction_proba[prediction] * 100
            
            # Display prediction with custom styling
            st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1.2rem;">Predicted Species</div>
                    <div class="prediction-result">🌸 {species.upper()}</div>
                    <div class="confidence-score">Confidence: {confidence:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution
            st.subheader("📈 Probability Distribution")
            proba_df = pd.DataFrame({
                'Species': metadata['target_names'],
                'Probability': prediction_proba
            })
            st.bar_chart(proba_df.set_index('Species'))
            
            # Additional info
            if confidence > 90:
                st.success("✅ Very confident prediction!")
            elif confidence > 70:
                st.info("ℹ️ Moderately confident prediction")
            else:
                st.warning("⚠️ Low confidence - model is uncertain")
        else:
            st.info("👈 Click the predict button to see results!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Built with ❤️ using Streamlit and Scikit-learn</p>
            <p>SVM Model for Iris Classification</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
