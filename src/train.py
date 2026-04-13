import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import mlflow
import mlflow.sklearn

DATA_DIR = Path(__file__).parent.parent / 'data'
MODELS_DIR = Path(__file__).parent.parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)     # Create models/ folder if it doesn't exist

def load_clean_data():
    df = pd.read_csv(DATA_DIR / 'heart_disease_clean.csv')
    
    # Features: everything except target and source
    X = df.drop(columns=['target', 'source'])
    y = df['target']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Scale the features (standardize to mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    best_model = None
    best_auc = 0
    
    for name, model in models.items():
        X_tr = X_train_scaled if name == 'Logistic Regression' else X_train
        X_te = X_test_scaled if name == 'Logistic Regression' else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n── {name} ──")
        print(f"  Accuracy : {acc*100:.1f}%")
        print(f"  ROC-AUC  : {auc:.3f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        if auc > best_auc:
            best_auc   = auc
            best_model = (name, model, scaler)

    return best_model

def main():
    print("Loading clean data...")
    X_train, X_test, y_train, y_test = load_clean_data()
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Set the experiment name — creates it if it doesn't exist
    mlflow.set_experiment("heart-disease-prediction")

    print("\nTraining models...")
    best_name, best_model, scaler = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    print(f"\n✅ Best model: {best_name}")

    # Log everything to MLflow
    with mlflow.start_run(run_name=best_name):
        # Log model parameters
        mlflow.log_param("model", best_name)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Get scores to log
        X_te = X_test
        if best_name == "Logistic Regression":
            X_te = scaler.transform(X_test)

        y_pred = best_model.predict(X_te)
        y_prob = best_model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Log metrics
        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("roc_auc",  round(auc, 4))

        # Log the model itself
        mlflow.sklearn.log_model(best_model, "model")

        print(f"  Logged to MLflow — accuracy: {acc*100:.1f}% | AUC: {auc:.3f}")

    # Save locally as before
    joblib.dump(best_model, MODELS_DIR / 'model.pkl')
    joblib.dump(scaler,     MODELS_DIR / 'scaler.pkl')
    print(f"  Model saved to models/model.pkl")
    print(f"  Scaler saved to models/scaler.pkl")

if __name__ == '__main__':
    main()