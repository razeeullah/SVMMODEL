# 🌸 Iris Flower Classification with SVM

A machine learning project that uses Support Vector Machine (SVM) to classify Iris flowers into three species, with an interactive Streamlit web application for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

This project demonstrates a complete machine learning pipeline:
1. **Data Loading**: Using the classic Iris dataset
2. **Model Training**: Support Vector Machine with RBF kernel
3. **Model Evaluation**: Comprehensive metrics and cross-validation
4. **Web Application**: Interactive Streamlit interface for predictions
5. **Deployment Ready**: Configured for GitHub and Streamlit Cloud

## ✨ Features

- ✅ **Accurate SVM Model**: Trained on the Iris dataset with high accuracy
- 🎨 **Beautiful UI**: Modern, responsive Streamlit interface
- 📊 **Real-time Predictions**: Instant classification with probability scores
- 📈 **Visualization**: Interactive charts showing prediction confidence
- 🔧 **User-Friendly**: Simple sliders for input features
- 🚀 **Production Ready**: Easy deployment to Streamlit Cloud

## 🌺 Dataset

The **Iris Dataset** is a classic dataset in machine learning, containing:
- **150 samples** of iris flowers
- **4 features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **3 classes**: Setosa, Versicolor, Virginica

### Feature Ranges:
- Sepal Length: 4.0 - 8.0 cm
- Sepal Width: 2.0 - 4.5 cm
- Petal Length: 1.0 - 7.0 cm
- Petal Width: 0.1 - 2.5 cm

## 📊 Model Performance

Our SVM model achieves excellent performance metrics:

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~97-100% |
| **Training Accuracy** | ~98-100% |
| **Cross-Validation Score** | ~96-98% |

### Confusion Matrix
The model shows excellent separation between classes with minimal misclassifications.

### Classification Report
- **Precision**: High across all classes
- **Recall**: Consistent performance
- **F1-Score**: Balanced precision and recall

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/iris-svm-classifier.git
cd iris-svm-classifier
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

Run the training script to create the SVM model:

```bash
python train_model.py
```

This will:
- Load the Iris dataset
- Preprocess the data
- Train the SVM model
- Display evaluation metrics
- Save model artifacts (`svm_model.pkl`, `scaler.pkl`, `model_metadata.pkl`)

### Running the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Adjust Input Features**: Use the sliders in the sidebar to set flower measurements
2. **Click Predict**: Press the "🔮 Predict Species" button
3. **View Results**: See the predicted species, confidence score, and probability distribution

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit: SVM Iris Classifier"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live** at: `https://your-app-name.streamlit.app`

### Deploy to Other Platforms

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Deploy
git push heroku main
```

#### Google Cloud Run
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT-ID/iris-app

# Deploy
gcloud run deploy --image gcr.io/PROJECT-ID/iris-app --platform managed
```

## 📁 Project Structure

```
iris-svm-classifier/
│
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
│
├── svm_model.pkl          # Trained SVM model (generated)
├── scaler.pkl             # Feature scaler (generated)
└── model_metadata.pkl     # Model metadata (generated)
```

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **Scikit-learn**: Machine learning library
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Joblib**: Model serialization

## 🔬 How It Works

### 1. Data Preprocessing
- Load Iris dataset
- Split into train/test sets (80/20)
- Scale features using StandardScaler

### 2. Model Training
- Algorithm: Support Vector Machine (SVM)
- Kernel: Radial Basis Function (RBF)
- Cross-validation: 5-fold

### 3. Prediction Pipeline
- User inputs → Feature scaling → SVM prediction → Display results

## 📈 Future Enhancements

- [ ] Add more classifiers (Random Forest, Neural Networks)
- [ ] Model comparison dashboard
- [ ] Feature importance visualization
- [ ] Support for custom datasets
- [ ] API endpoint for predictions
- [ ] Docker containerization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created with ❤️ for machine learning education

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Iris dataset
- Scikit-learn community
- Streamlit team for the amazing framework

---

**⭐ If you found this project helpful, please give it a star!**

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.
