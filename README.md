# Automobile Valuator Engine: Production Machine Learning Architecture

A production-grade, end-to-end predictive machine learning deployment system. The project features an optimized inference engine, an asynchronous containerized FastAPI backend, and a cross-platform Flutter application structured under a strict MVVM pattern.

## System Topology Blueprint

```text
linear_regression_model/
│
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb       # Structural Research & Modeling Log
│   │   ├── dataset/
│   │   │   └── used_cars.csv        # Dynamic Master CSV Analytics Datastore
│   │   ├── models/
│   │   │   └── production_artifacts.joblib # Unified Transform Pipeline Data
│   │   ├── plots/
│   │   └── prediction.py            # Local CLI Testing Engine Script
│   │
│   ├── API/
│   │   ├── main.py                  # App Entrypoint & Route Orchestrator
│   │   ├── schemas.py               # Pydantic Structural Validation Layer
│   │   ├── prediction.py            # Live Pipeline Decompression Script
│   │   ├── retrain.py               # Asynchronous Shadow Evaluation Logic
│   │   ├── requirements.txt
│   │   └── render.yaml
│   │
│   ├── FlutterApp/
│   │   ├── lib/
│   │   │   ├── constants/
│   │   │   ├── models/
│   │   │   ├── services/
│   │   │   ├── widgets/
│   │   │   ├── screens/
│   │   │   └── main.dart
│   │   └── pubspec.yaml
│   
├── README.md
├── pyproject.toml
└── uv.lock