# Used Car Asset Valuation Engine

A full-stack, machine learning-powered mobile application and RESTful API that predicts used car market valuations in real time. Built with Flutter, FastAPI, scikit-learn, and deployed on Render.

---

## Key Features

* Real-Time Predictive Inference: Connects a Flutter mobile UI to a remote production ML pipeline over HTTP.
* Scikit-Learn ML Pipeline: Leverages a trained LinearRegression model bundled with StandardScaler and OneHotEncoder artifacts.
* Deterministic Feature Alignment: Converts raw user inputs (Year, Brand, Fuel Type, etc.) into exact 67-column encoded feature matrices automatically.
* Automated Data Preprocessing: Dynamically calculates derivative metrics like Car Age from user-selected model years.
* CI/CD & Cloud Deployment: Continuous deployment pipeline hosted on Render with instant hot-reloading capability.

---

## Tech Stack

### Mobile Frontend
* Framework: Flutter (Dart)
* Networking: http package for REST communications
* UI/UX: Material Design interface with interactive feedback states

### Backend API
* Framework: FastAPI (Python)
* ASGI Server: Uvicorn
* Data Processing & ML: pandas, numpy, scikit-learn, joblib
* Deployment: Render Web Service

---

## Getting Started Locally

### 1. Prerequisites
* Flutter SDK installed locally.
* Python 3.10 or higher installed.

---

### 2. Backend Setup (FastAPI)

1. Clone the repository and navigate into the project directory.

2. Create and activate a virtual environment:
   python -m venv venv
   
   On Windows:
   .\venv\Scripts\activate
   
   On macOS/Linux:
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Run the local API server:
   python -m uvicorn summative.API.main:app --reload --host 0.0.0.0 --port 8000
   
   The API will be available at http://localhost:8000 and interactive Swagger Documentation at http://localhost:8000/docs.

---

### 3. Mobile App Setup (Flutter)

1. Navigate to your Flutter project directory.

2. Install Flutter packages:
   flutter pub get

3. Run the app on an emulator or physical device:
   flutter run

---

## API Endpoint Reference

### POST /predict

Calculates the market valuation for a given vehicle specification.

#### Request Body Sample:
```json
{
  "manufacturer_brand": "Toyota",
  "model_year": 2018,
  "odometer": 65000,
  "combustible_fuel_type": "Petrol",
  "accident": "None reported"
}
