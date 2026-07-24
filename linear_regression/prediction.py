"""
Local Standalone CLI Inference Validation Script.
Allows quick testing of the ML model independently of network layers.
"""

import sys
from typing import Dict, Any
from datetime import datetime

# Direct module routing adjustment
try:
    from summative.API.prediction import run_inference
except ImportError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from summative.API.prediction import run_inference

def main() -> None:
    # Diagnostic test mock tracking real feature dimensions
    mock_payload: Dict[str, Any] = {
        "year": 2021,
        "mileage": 24500.0,
        "engine_size": 2.0,
        "hp": 252.0,
        "brand": "BMW",
        "transmission": "Automatic",
        "fuel_type": "Petrol"
    }
    
    print(f"[{datetime.now().isoformat()}] Simulating local prediction routine...")
    print(f"Feature vector inputs:\n {mock_payload}")
    
    try:
        valuation = run_inference(mock_payload)
        print("\n=== INFERENCE EXECUTION SUCCESS ===")
        print(f"Calculated Asset Market Valuation: ${valuation:,.2f}")
        print("====================================")
    except FileNotFoundError:
        print("\n[ERROR] Production artifacts not detected.")
        print("Run the data analysis notebook or retraining pipeline to generate models.")
    except Exception as error:
        print(f"\n[CRITICAL CRASH] Execution aborted: {str(error)}")

if __name__ == "__main__":
    main()