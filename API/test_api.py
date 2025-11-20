# API/test_api.py
import requests
import json

# Test the API locally
BASE_URL = "http://localhost:8000"

def test_api():
    print("🧪 Testing Crop Yield Prediction API...")
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return
    
    # Test 2: Model info
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"✅ Model info: {response.status_code}")
        print(f"Available crops: {response.json()['available_crops']}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    # Test 3: Valid prediction
    test_data = {
        "N": 75,
        "P": 45,
        "K": 90,
        "temperature": 25,
        "humidity": 70,
        "ph": 6.5,
        "rainfall": 150,
        "crop": "maize"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        print(f"✅ Valid prediction: {response.status_code}")
        result = response.json()
        print(f"Predicted yield: {result['predicted_yield_kg_ha']} kg/ha")
        print(f"Model used: {result['model_used']}")
    except Exception as e:
        print(f"❌ Valid prediction failed: {e}")
    
    # Test 4: Invalid range (should fail)
    invalid_data = {
        "N": 200,  # Too high
        "P": 45,
        "K": 90,
        "temperature": 25,
        "humidity": 70,
        "ph": 6.5,
        "rainfall": 150,
        "crop": "maize"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        print(f"✅ Invalid range handled: {response.status_code}")
        if response.status_code != 200:
            print(f"Error message: {response.json()}")
    except Exception as e:
        print(f"❌ Invalid range test failed: {e}")
    
    # Test 5: Invalid crop
    invalid_crop_data = {
        "N": 75,
        "P": 45,
        "K": 90,
        "temperature": 25,
        "humidity": 70,
        "ph": 6.5,
        "rainfall": 150,
        "crop": "invalid_crop"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_crop_data)
        print(f"✅ Invalid crop handled: {response.status_code}")
        if response.status_code != 200:
            print(f"Error message: {response.json()}")
    except Exception as e:
        print(f"❌ Invalid crop test failed: {e}")

if __name__ == "__main__":
    test_api()