import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate():
    data_path = 'data/preprocessed_data.csv'
    model_dir = 'model'
    static_img_dir = 'static/img'
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)
    
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Feature columns and target
    X = df[['Volatility', 'Trend', 'Volume_Norm']]
    y = df['Risk_Class']
    
    # Ensure no data leakage: Split BEFORE Scaling
    logging.info("Performing 80/20 train/test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    logging.info("Scaling features using StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved scaler to {scaler_path}")
    
    # Train the MLPClassifier
    logging.info("Training MLPClassifier")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    logging.info("Evaluating model")
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall: {rec:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    
    # Cross Validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    logging.info(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    # Save Model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    logging.info(f"Saved model to {model_path}")
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title('Neural Network Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    cm_path = os.path.join(static_img_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Saved Confusion Matrix plot to {cm_path}")

if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        logging.error(f"Error during training: {e}")
