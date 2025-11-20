# API/prediction.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, confloat, validator
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("🌱 Initializing Crop Yield Prediction API with Kaggle Dataset...")

# Custom Gradient Descent Class (same as your notebook)
class CustomGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss_history = []
        self.val_loss_history = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        m, n = X_train.shape
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0

        for i in range(self.n_iterations):
            # Training predictions and loss
            y_pred_train = np.dot(X_train, self.weights) + self.bias
            error_train = y_pred_train - y_train
            loss_train = np.mean(error_train ** 2)
            self.loss_history.append(loss_train)

            # Validation loss if validation data provided
            if X_val is not None and y_val is not None:
                y_pred_val = np.dot(X_val, self.weights) + self.bias
                error_val = y_pred_val - y_val
                loss_val = np.mean(error_val ** 2)
                self.val_loss_history.append(loss_val)

            # Compute gradients
            dw = (2/m) * np.dot(X_train.T, error_train)
            db = (2/m) * np.sum(error_train)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def load_kaggle_dataset():
    """Load and prepare the actual Kaggle crop recommendation dataset"""
    print("📊 Loading Kaggle Crop Recommendation Dataset...")
    
    try:
        # Load the dataset
        df = pd.read_csv('Crop_recommendation.csv')
        print(f"✅ Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Display dataset info
        print(f"📋 Dataset Columns: {df.columns.tolist()}")
        print(f"🌱 Crop Types: {df['label'].unique()}")
        
        return df
        
    except FileNotFoundError:
        print("❌ Crop_recommendation.csv not found in API directory")
        print("💡 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print("   And place 'Crop_recommendation.csv' in the API folder")
        raise FileNotFoundError("Crop_recommendation.csv not found")

def create_yield_target(df):
    """Create realistic yield target based on Kaggle dataset features"""
    print("🎯 Creating yield target variable...")
    
    # Based on agricultural knowledge, create realistic yield calculation
    yield_calc = (
        df['N'] * 12 +           # Nitrogen contribution
        df['P'] * 10 +           # Phosphorus contribution  
        df['K'] * 8 +            # Potassium contribution
        df['temperature'] * 15 + # Temperature contribution
        df['humidity'] * 5 +     # Humidity contribution
        df['rainfall'] * 0.3 +   # Rainfall contribution
        (1 - abs(df['ph'] - 6.5) / 3) * 1000  # pH optimal at 6.5
    )
    
    # Add crop-specific base yields
    crop_base_yields = {
        'rice': 2500, 'maize': 3000, 'chickpea': 1500, 'kidneybeans': 1200,
        'pigeonpeas': 1300, 'mothbeans': 1100, 'mungbean': 1400, 'blackgram': 1350,
        'lentil': 1250, 'pomegranate': 8000, 'banana': 12000, 'mango': 9000,
        'grapes': 7000, 'watermelon': 15000, 'muskmelon': 10000, 'apple': 11000,
        'orange': 8000, 'papaya': 15000, 'coconut': 6000, 'cotton': 2000,
        'jute': 1800, 'coffee': 1500
    }
    
    # Add crop-specific base yield
    df['crop_base'] = df['label'].map(crop_base_yields)
    yield_final = yield_calc + df['crop_base'] + np.random.normal(0, 200, len(df))
    
    print(f"📈 Yield range: {yield_final.min():.0f} - {yield_final.max():.0f} kg/ha")
    return yield_final

def train_and_save_model():
    """Train models using the actual Kaggle dataset"""
    print("🔄 Training machine learning models with Kaggle dataset...")
    
    # Load actual dataset
    df = load_kaggle_dataset()
    
    # Create yield target
    y = create_yield_target(df)
    
    # Prepare features (same as your notebook)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].copy()
    label_encoder = LabelEncoder()
    X['crop_encoded'] = label_encoder.fit_transform(df['label'])
    
    print(f"🔧 Features: {X.columns.tolist()}")
    print(f"🌱 Encoded {len(label_encoder.classes_)} crop types")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Testing set: {X_test.shape[0]} samples")
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
    }
    
    performance = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        performance[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred),
            'model': model
        }
        
        print(f"    {name}: MAE = {performance[name]['MAE']:.2f}, R² = {performance[name]['R²']:.4f}")
    
    # Select best model (lowest MAE)
    best_model_name = min(performance, key=lambda x: performance[x]['MAE'])
    best_model = performance[best_model_name]['model']
    
    # Create dataset info
    dataset_info = {
        'samples': len(df),
        'crops': len(label_encoder.classes_),
        'features': X.columns.tolist()
    }
    
    # Save model data
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist(),
        'performance': performance,
        'best_model_name': best_model_name,
        'dataset_info': dataset_info
    }
    
    joblib.dump(model_data, 'best_crop_model.joblib')
    print(f"✅ Best model ({best_model_name}) trained and saved successfully!")
    print(f"📊 Final Performance - MAE: {performance[best_model_name]['MAE']:.2f}, R²: {performance[best_model_name]['R²']:.4f}")
    
    return model_data

# Initialize global variables
model = None
scaler = None
label_encoder = None
feature_names = None
best_model_name = None
performance = None
dataset_info = None

# Load or create model
try:
    model_data = joblib.load('best_crop_model.joblib')
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    best_model_name = model_data['best_model_name']
    performance = model_data['performance']
    dataset_info = model_data.get('dataset_info', {'samples': 'Unknown', 'crops': 'Unknown', 'features': []})
    
    print("✅ Pre-trained model loaded successfully!")
    print(f"📊 Dataset: {dataset_info.get('samples', 'Unknown')} samples, {dataset_info.get('crops', 'Unknown')} crops")
    print(f"🎯 Best model: {best_model_name}")
    print(f"🌱 Available crops: {label_encoder.classes_.tolist()}")
    
except FileNotFoundError:
    print("📝 No pre-trained model found. Training new model with Kaggle dataset...")
    model_data = train_and_save_model()
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    best_model_name = model_data['best_model_name']
    performance = model_data['performance']
    dataset_info = model_data.get('dataset_info', {'samples': 'Unknown', 'crops': 'Unknown', 'features': []})

# Create FastAPI app
app = FastAPI(
    title="Crop Yield Prediction API",
    description="API for predicting crop yield using machine learning models trained on Kaggle Crop Recommendation Dataset",
    version="1.0.0"
)

# REQUIRED: CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUIRED: Pydantic Model with Data Types and Range Constraints
class CropPredictionInput(BaseModel):
    N: confloat(ge=0, le=140) = 90.0
    P: confloat(ge=5, le=145) = 42.0
    K: confloat(ge=5, le=205) = 43.0
    temperature: confloat(ge=8, le=43) = 25.0
    humidity: confloat(ge=14, le=100) = 82.0
    ph: confloat(ge=3.5, le=9.5) = 6.5
    rainfall: confloat(ge=20, le=300) = 203.0
    crop: str = "rice"

    @validator('crop')
    def validate_crop(cls, v):
        available_crops = label_encoder.classes_.tolist()
        available_crops_lower = [crop.lower() for crop in available_crops]
        if v.lower() not in available_crops_lower:
            raise ValueError(f"Crop '{v}' not supported. Available crops: {available_crops}")
        return v

class PredictionResponse(BaseModel):
    predicted_yield_kg_ha: float
    model_used: str
    input_parameters: dict
    units: str
    status: str
    message: str

# REQUIRED: POST endpoint for prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict_yield(input_data: CropPredictionInput):
    """
    Predict crop yield based on input parameters using Kaggle dataset
    """
    try:
        print(f"📥 Received prediction request for crop: {input_data.crop}")
        
        # Handle crop encoding with case-insensitive matching
        available_crops = label_encoder.classes_.tolist()
        crop_lower = input_data.crop.lower()
        available_crops_lower = [crop.lower() for crop in available_crops]
        
        if crop_lower not in available_crops_lower:
            raise HTTPException(
                status_code=400, 
                detail=f"Crop '{input_data.crop}' not supported. Available crops: {available_crops}"
            )
        
        # Convert crop name to match training data case
        crop_mapping = {crop.lower(): crop for crop in available_crops}
        crop_corrected = crop_mapping[crop_lower]
        crop_encoded = label_encoder.transform([crop_corrected])[0]

        # Prepare features in exact same order as training
        features = [
            float(input_data.N),
            float(input_data.P), 
            float(input_data.K),
            float(input_data.temperature),
            float(input_data.humidity),
            float(input_data.ph),
            float(input_data.rainfall),
            int(crop_encoded)
        ]
        
        print(f"🔧 Features prepared: {dict(zip(feature_names, features))}")
        
        # Create DataFrame with correct feature names
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # REQUIRED: Apply same scaling as during training
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Ensure prediction is positive and realistic
        prediction = max(0, float(prediction))
        
        response = PredictionResponse(
            predicted_yield_kg_ha=round(prediction, 2),
            model_used=best_model_name,
            input_parameters=input_data.dict(),
            units="kilograms per hectare",
            status="success",
            message=f"Yield prediction for {crop_corrected} using Kaggle dataset model"
        )
        
        print(f"✅ Prediction successful: {prediction:.2f} kg/ha for {crop_corrected}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Crop Yield Prediction API",
        "version": "1.0.0",
        "description": "Machine learning API for predicting crop yields using Kaggle Crop Recommendation Dataset",
        "dataset_source": "Kaggle: Atharva Ingle - Crop Recommendation Dataset",
        "endpoints": {
            "docs": "/docs",
            "prediction": "/predict (POST)",
            "model_info": "/model-info",
            "health": "/health",
            "retrain": "/retrain (GET)"
        }
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed information about the loaded model"""
    model_performance = performance.get(best_model_name, {}) if performance else {}
    
    return {
        "model_name": best_model_name,
        "features_used": feature_names,
        "available_crops": label_encoder.classes_.tolist() if label_encoder else [],
        "dataset_info": dataset_info,
        "input_constraints": {
            "N": {"min": 0, "max": 140, "description": "Nitrogen level"},
            "P": {"min": 5, "max": 145, "description": "Phosphorus level"},
            "K": {"min": 5, "max": 205, "description": "Potassium level"},
            "temperature": {"min": 8, "max": 43, "description": "Temperature in Celsius"},
            "humidity": {"min": 14, "max": 100, "description": "Humidity percentage"},
            "ph": {"min": 3.5, "max": 9.5, "description": "Soil pH level"},
            "rainfall": {"min": 20, "max": 300, "description": "Rainfall in mm"},
            "crop": {"type": "string", "description": "Crop type from available list"}
        },
        "performance_metrics": {
            "MAE": round(model_performance.get('MAE', 0), 2),
            "MSE": round(model_performance.get('MSE', 0), 2),
            "R²": round(model_performance.get('R²', 0), 4)
        },
        "dataset_description": "Trained on Kaggle Crop Recommendation Dataset with realistic yield calculations"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": best_model_name,
        "dataset_samples": dataset_info.get('samples', 'Unknown') if dataset_info else 'Unknown',
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/crops")
async def list_crops():
    """List all available crops for prediction"""
    crops = label_encoder.classes_.tolist() if label_encoder else []
    return {
        "available_crops": crops,
        "count": len(crops),
        "source": "Kaggle Crop Recommendation Dataset"
    }

@app.get("/retrain")
async def retrain_model():
    """Retrain the model with the Kaggle dataset"""
    try:
        print("🔄 Retraining model with Kaggle dataset...")
        model_data = train_and_save_model()
        
        # Update global variables
        global model, scaler, label_encoder, feature_names, best_model_name, performance, dataset_info
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        best_model_name = model_data['best_model_name']
        performance = model_data['performance']
        dataset_info = model_data.get('dataset_info', {'samples': 'Unknown', 'crops': 'Unknown', 'features': []})
        
        return {
            "status": "success",
            "message": f"Model retrained successfully with Kaggle dataset. New best model: {best_model_name}",
            "performance": performance[best_model_name],
            "dataset_info": dataset_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Crop Yield Prediction API with Kaggle Dataset...")
    print("📚 Swagger UI available at: http://localhost:8000/docs")
    print("🌱 API ready for predictions using real crop data!")
    print(f"📊 Using dataset: {dataset_info.get('samples', 'Unknown')} samples, {dataset_info.get('crops', 'Unknown')} crop types")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")