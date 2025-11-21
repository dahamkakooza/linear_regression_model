# Crop Yield Prediction System

## Mission & Problem
**Agricultural Optimization**: Predict crop yields using machine learning to help farmers optimize resources and improve food security. Addresses real-world agricultural challenges through data-driven insights.

**Dataset**: Kaggle Crop Recommendation (2,200 samples, 22 crops, 8 features)  
**ML Task**: Regression analysis predicting yield in kg/ha

## 🚀 Quick Access
- **Live API**: https://crop-yield-api-pfsb.onrender.com
- **API Docs**: https://crop-yield-api-pfsb.onrender.com/docs  
- **Mobile App**: Flutter (Android/iOS) - See FlutterApp/
- **Best Model**: Random Forest (R²=0.969, MAE=291.81)

## 📊 System Architecture
linear_regression_model/
├── summative/linear_regression/multivariate.ipynb # ML Models
├── summative/API/prediction.py # FastAPI Backend
├── summative/FlutterApp/ # Mobile Frontend
└── Deployment on Render.com

text

## 🎯 Key Achievements

### Task 1: Machine Learning
- **3 Models Trained**: Linear Regression, Decision Tree, Random Forest
- **Best Performance**: Random Forest (97% variance explained)
- **Visualizations**: Correlation heatmaps, distribution plots, loss curves
- **Feature Engineering**: Realistic yield calculation from agricultural data

### Task 2: Production API
- **FastAPI** with automatic Swagger documentation
- **Pydantic Validation**: Type safety & range constraints (N:0-140, P:5-145, etc.)
- **CORS Enabled**: Cross-origin support for mobile app
- **Deployed**: Publicly available on Render.com

### Task 3: Mobile Application  
- **Flutter Cross-platform**: Android, iOS, Web
- **8 Input Fields**: Matches API specification exactly
- **Real-time Predictions**: Live API integration
- **Input Validation**: Comprehensive error handling

## 📈 Model Performance
| Model | R² Score | MAE | Status |
|-------|----------|-----|--------|
| Random Forest | 0.9692 | 291.81 | ✅ **Production** |
| Decision Tree | 0.9528 | 370.57 |  |
| Linear Regression | 0.5653 | 2700.73 |  |

## 🛠️ Tech Stack
- **Backend**: Python, FastAPI, Scikit-learn, Pandas
- **Frontend**: Flutter/Dart, Material Design
- **Deployment**: Render.com
- **Data**: Kaggle Agricultural Dataset

## 🎓 Academic Compliance
✅ Non-generic agricultural use case  
✅ Rich dataset with proper sourcing  
✅ Three ML models with comparison  
✅ Production API with validation  
✅ Mobile app with required features  
✅ Comprehensive documentation  

---
*Built for educational purposes - Demonstrating full-stack ML deployment*
