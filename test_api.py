import requests

# The API address
URL = "http://127.0.0.1:8000/predict"

# Test case 1 — likely HEALTHY patient
healthy_patient = {
    "age":      45.0,
    "sex":       0.0,   # female
    "cp":        2.0,   # atypical angina
    "trestbps": 120.0,
    "chol":     230.0,
    "fbs":        0.0,
    "restecg":    0.0,
    "thalach":  170.0,
    "exang":      0.0,
    "oldpeak":    0.2,
    "slope":      1.0,
    "ca":         0.0,
    "thal":       3.0
}

# Test case 2 — likely DISEASE patient
disease_patient = {
    "age":      63.0,
    "sex":       1.0,   # male
    "cp":        4.0,   # asymptomatic
    "trestbps": 145.0,
    "chol":     233.0,
    "fbs":        1.0,
    "restecg":    2.0,
    "thalach":  150.0,
    "exang":      0.0,
    "oldpeak":    2.3,
    "slope":      3.0,
    "ca":         0.0,
    "thal":       6.0
}

def test_patient(name, data):
    response = requests.post(URL, json=data)
    result = response.json()
    print(f"\n── {name} ──")
    print(f"  Diagnosis   : {result['diagnosis']}")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Probability : {result['probability']}")

if __name__ == '__main__':
    print("Testing Heart Disease Prediction API...\n")
    test_patient("Healthy Patient", healthy_patient)
    test_patient("Disease Patient", disease_patient)