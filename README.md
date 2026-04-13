# Heart Disease MLOps

A production-ready machine learning pipeline to predict heart disease,
built with scikit-learn, FastAPI, MLflow, and Docker.

## Dataset
- **Source:** UCI Heart Disease Dataset
- **4 hospitals:** Cleveland, Hungarian, Switzerland, VA Long Beach
- **920 patients** total | **13 features** | Binary target (0=Healthy, 1=Disease)

## Project Structure
```
heart-disease-mlops/
├── data/ # Raw and cleaned datasets
├── src/
│ ├── preprocess.py # Data cleaning & merging
│ ├── train.py # Model training + MLflow tracking
│ └── api.py # FastAPI prediction API
├── models/ # Saved trained models
├── test_api.py # API tests
├── Dockerfile # Container definition
├── docker-compose.yml # Container orchestration
└── requirements.txt # Dependencies
```

## How to Run

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocess data
```bash
python src/preprocess.py
```

### 3. Train model
```bash
python src/train.py
```

### 4. Start API
```bash
uvicorn src.api:app --reload
```

### 5. Run with Docker
```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/`        | Health check |
| POST | `/predict` | Predict heart disease |

**Example request:**
```json
{
  "age": 63.0, "sex": 1.0, "cp": 4.0,
  "trestbps": 145.0, "chol": 233.0, "fbs": 1.0,
  "restecg": 2.0, "thalach": 150.0, "exang": 0.0,
  "oldpeak": 2.3, "slope": 3.0, "ca": 0.0, "thal": 6.0
}
```

**Example response:**
```json
{
  "prediction": 1,
  "diagnosis": "Heart Disease",
  "probability": 0.821
}
```

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~83% | ~0.90 |
| **Random Forest** ✅ | **~83%** | **0.919** |

## Tech Stack
- **ML:** scikit-learn
- **API:** FastAPI + Uvicorn
- **Tracking:** MLflow
- **Container:** Docker
- **Language:** Python 3.11



