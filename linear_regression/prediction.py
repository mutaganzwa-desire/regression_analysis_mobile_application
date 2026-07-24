"""
Local Standalone CLI Inference Validation Script.
Allows quick testing of the ML model independently of network layers.
"""

import sys
from typing import Dict, Any
from datetime import datetime, timezone

# Direct module routing adjustment
try:
    from summative.API.prediction import run_inference
except ImportError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from summative.API.prediction import run_inference

def main() -> None:
    # Payload matching real dataset feature specifications
    mock_payload: Dict[str, Any] = {
        "brand": "Ford",
        "model_year": 2018,
        "milage": 45000.0,
        "fuel_type": "Gasoline",
        "accident": "None reported"
    }
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Simulating local prediction routine...")
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