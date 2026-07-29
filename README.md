# Used Car Asset Valuation Engine

## Description of Mission and Problem
Used car buyers and sellers struggle with inaccurate vehicle price estimations due to fragmented market data and complex feature interactions. 
This project provides a machine learning engine that calculates real-time vehicle valuations using physical and operational metrics. It exposes a public FastAPI backend integrated with a Flutter mobile app for instant asset appraisal.

---

## Publicly Available API Endpoint

* Production API Base URL: [https://used-car-pricing-api.onrender.com](https://used-car-pricing-api.onrender.com)
* Swagger UI Testing Interface: [https://used-car-pricing-api.onrender.com/docs](https://used-car-pricing-api.onrender.com/docs)

Note: The API endpoint uses a publicly routable HTTPS URL hosted on Render and can be tested directly using the interactive Swagger UI documentation at the `/docs` endpoint.

---

## Video Demonstration

A 7-minute video demonstration covering the machine learning model development, backend FastAPI deployment, and live Flutter mobile application workflow is available on YouTube:

* Video Demo Link: [https://youtu.be/sRVLhcZu7IE](https://youtu.be/sRVLhcZu7IE)

---

## Instructions to Run the Mobile Application

### Prerequisites
* Flutter SDK (Version 3.0.0 or higher) installed.
* An active Android Emulator, iOS Simulator, or connected physical testing device.

### Execution Steps

1. Open a terminal and navigate to the Flutter application directory:
   cd linear_regression_model\summative\FlutterApp

2. Fetch required package dependencies:
   flutter pub get

3. Verify connected devices:
   flutter devices

4. Run the application on your target device:
   flutter run

---

## API Request and Response Specification

### POST /predict

Accepts vehicle parameters and returns the predicted market valuation.

#### Sample Request Body:
```json
{
  "manufacturer_brand": "Toyota",
  "model_year": 2018,
  "odometer": 65000,
  "combustible_fuel_type": "Petrol",
  "accident": "None reported"
}
