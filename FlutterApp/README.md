# Crop Yield Predictor - Flutter Mobile App

## 📱 Mobile Application for Agricultural Predictions

**Platform**: Android, iOS, Web  
**API Integration**: Live ML backend at https://crop-yield-api-pfsb.onrender.com

## 🎯 Core Features

### Input Management
- **8 Parameter Fields**: N, P, K, Temperature, Humidity, pH, Rainfall, Crop Type
- **Real-time Validation**: Range checking & numeric input validation
- **Pre-filled Samples**: Realistic agricultural values for quick testing

### Prediction Engine
- **Live API Calls**: Connects to deployed Random Forest model
- **Loading States**: Visual feedback during prediction
- **Result Display**: Clear yield predictions in kg/ha

### User Experience  
- **Material Design**: Professional, intuitive interface
- **Error Handling**: Comprehensive validation messages
- **Cross-platform**: Single codebase for all devices

## 🛠️ Technical Implementation

### Architecture
```dart
lib/main.dart
├── CropYieldPredictorApp()      // MaterialApp root
├── PredictionPage()             // Main screen
│   ├── _PredictionPageState()   // Business logic
│   │   ├── predictYield()       // API communication
│   │   ├── _validateInputs()    // Client-side validation
│   │   └── _buildInputField()   // UI components
