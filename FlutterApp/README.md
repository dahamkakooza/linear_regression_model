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
API Integration
dart
// Endpoint configuration
final String apiUrl = "https://crop-yield-api-pfsb.onrender.com/predict";

// Request format
Map<String, dynamic> requestBody = {
  "N": 90, "P": 42, "K": 43,           // Soil nutrients
  "temperature": 25, "humidity": 82,   // Environment  
  "ph": 6.5, "rainfall": 203,          // Soil conditions
  "crop": "rice"                       // Crop selection
};
Validation Rules
Parameter	Range	Validation
Nitrogen (N)	0-140	Numeric, range check
Phosphorus (P)	5-145	Numeric, range check
Potassium (K)	5-205	Numeric, range check
Temperature	8-43°C	Numeric, range check
Humidity	14-100%	Numeric, range check
pH Level	3.5-9.5	Numeric, range check
Rainfall	20-300mm	Numeric, range check
Crop Type	22 options	Dropdown selection
🚀 Getting Started
Prerequisites
Flutter SDK 3.0+

Android Studio / VS Code with Flutter extension

Android Emulator or physical device

Installation & Run
bash
cd FlutterApp
flutter pub get        # Install dependencies
flutter run           # Launch on connected device
For Web Demo
bash
flutter create --platforms web .  # Enable web support
flutter run -d chrome             # Run in browser
📋 Supported Crops
22 agricultural types including: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

🎨 UI/UX Design
Layout Structure
AppBar: Branding with green agricultural theme

Header Card: Application purpose and API status

Input Section: Organized parameter groups (Soil, Environment, Crop)

Action Button: Prominent prediction trigger

Result Display: Success state with yield information

Error States: Clear validation feedback

Visual Features
Material Design 3: Modern Flutter components

Color Scheme: Green theme representing agriculture

Responsive Layout: Adapts to different screen sizes

Loading Indicators: Circular progress during API calls

🔧 Development
Dependencies
yaml
dependencies:
  flutter: sdk:flutter
  http: ^1.1.0          # API communication
  cupertino_icons: ^1.0.2
Key Components
State Management: setState for local state

HTTP Client: Dart http package for REST calls

Form Validation: Custom validation logic

Error Handling: Try-catch with user feedback

🎥 Demo Scenario
Launch App: Opens with pre-filled sample data

Predict Rice: Click button → See ~6175 kg/ha result

Test Validation: Enter invalid values → See error messages

Change Crop: Select maize → Get different prediction

Range Testing: Test boundary values (min/max ranges)

✅ Rubric Compliance
Requirement	Implementation
8 input fields	Complete parameter set
Predict button	Large, labeled action button
Display area	Dedicated result card
Error handling	Comprehensive validation
Organized layout	Material Design grouping
API integration	Live backend connection
Professional Flutter application demonstrating mobile ML integration
