"""
SVM Classification Model Training Script
Dataset: Iris Dataset
Author: ML Project
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("SVM CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # 1. Load Dataset
    print("\n[1/6] Loading Iris Dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {len(target_names)} ({', '.join(target_names)})")
    
    # 2. Create DataFrame for better visualization
    print("\n[2/6] Creating dataset overview...")
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = df['target'].map({i: name for i, name in enumerate(target_names)})
    
    print("\nDataset Info:")
    print(df.describe())
    print(f"\nClass distribution:\n{df['species'].value_counts()}")
    
    # 3. Split Data
    print("\n[3/6] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    # 4. Feature Scaling
    print("\n[4/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled using StandardScaler")
    
    # 5. Train SVM Model
    print("\n[5/6] Training SVM model...")
    svm_model = SVC(
        kernel='rbf',        # Radial Basis Function kernel
        C=1.0,               # Regularization parameter
        gamma='scale',       # Kernel coefficient
        probability=True,    # Enable probability estimates
        random_state=42
    )
    
    svm_model.fit(X_train_scaled, y_train)
    print("✓ Model training completed!")
    
    # 6. Model Evaluation
    print("\n[6/6] Evaluating model performance...")
    print("-"*60)
    
    # Training accuracy
    y_train_pred = svm_model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Testing accuracy
    y_test_pred = svm_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📊 ACCURACY SCORES:")
    print(f"  - Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  - Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Cross-validation
    cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
    print(f"\n🔄 CROSS-VALIDATION (5-Fold):")
    print(f"  - Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Confusion Matrix
    print(f"\n📈 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Classification Report
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # 7. Save Model Artifacts
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS")
    print("="*60)
    
    joblib.dump(svm_model, 'svm_model.pkl')
    print("✓ Model saved: svm_model.pkl")
    
    joblib.dump(scaler, 'scaler.pkl')
    print("✓ Scaler saved: scaler.pkl")
    
    # Save metadata
    metadata = {
        'feature_names': list(feature_names),
        'target_names': list(target_names),
        'test_accuracy': float(test_accuracy),
        'train_accuracy': float(train_accuracy),
        'cv_mean_score': float(cv_scores.mean()),
        'cv_std_score': float(cv_scores.std())
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print("✓ Metadata saved: model_metadata.pkl")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()
