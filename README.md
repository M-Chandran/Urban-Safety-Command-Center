# Urban Safety Command Center 🏙️🔒

A Flask-based web application for crime data analysis, visualization, and machine learning predictions. This platform enables law enforcement and urban planners to analyze historical crime data, train predictive models, and forecast future crime trends.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey?style=flat&logo=flask)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?style=flat)

## ✨ Features

### 📊 Data Management
- Upload crime data files (CSV, XLSX, XLS)
- Automatic data parsing and cleaning
- Column detection and statistics analysis

### 📈 Visualization
- Interactive charts (Bar, Line, Pie, Doughnut, Radar, Polar Area)
- Top 3 high-risk area ranking
- Crime trend analysis with contextual impact

### 🤖 Machine Learning
Multiple regression algorithms available:
- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net (L1 + L2)
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Context-Aware Bayesian Tensor Decomposition (Custom)

### 🔮 Predictions
- Single-step crime predictions
- Multi-step forecasting (up to 30 days)
- Risk level assessment (CRITICAL, HIGH, MEDIUM, LOW, VERY LOW)
- Confidence scoring
- Prediction export to CSV

### 👥 User Management
- User registration and authentication
- Personal dashboard with prediction history
- Individual data and model storage

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   
```
bash
   git clone <repository-url>
   cd web
   
```

2. **Create a virtual environment (recommended)**
   
```
bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   
```

3. **Install dependencies**
   
```
bash
   pip install -r requirements.txt
   
```

4. **Run the application**
   
```
bash
   python app.py
   
```

5. **Open your browser**
   Navigate to `http://localhost:5000`

### Default Setup

- Register a new account to get started
- Upload your crime data files
- Train ML models and make predictions

## 📁 Project Structure

```
web/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/
│   └── style.css        # Frontend styling
└── templates/
    ├── login.html       # User login page
    ├── register.html   # User registration page
    └── dashboard.html  # Main dashboard interface
```

## ⚙️ Configuration

### Secret Key
Change the secret key in `app.py` for production:
```
python
app.secret_key = 'your-secure-secret-key-change-in-production'
```

### File Upload Settings
- Allowed extensions: `.csv`, `.xlsx`, `.xls`
- Upload folder: `uploads/`
- Models folder: `models/`
- History folder: `history/`

## 🔧 Technologies Used

### Backend
- **Flask** - Web framework
- **Flask-Login** - User authentication
- **Werkzeug** - Security utilities

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Machine Learning
- **Scikit-learn** - ML algorithms

### Frontend
- **HTML5/CSS3** - Structure and styling
- **Chart.js** - Interactive charts

## 📝 Usage Guide

### 1. Upload Crime Data
- Navigate to the dashboard
- Click "Upload Crime Data"
- Select a CSV or Excel file
- The system will analyze and display statistics

### 2. Train a Model
- Click "Train" on a uploaded file
- Choose an algorithm from the dropdown
- Select the target column to predict
- Give your model a name
- Click "Train Model"

### 3. Make Predictions
- Select a trained model
- Enter feature values (comma-separated)
- Click "Predict" for single predictions
- Or use "Multi-Step Prediction" for forecasting

### 4. Analyze Trends
- Click "Trends" on any uploaded file
- View statistics, trends, and contextual impact analysis

## ⚠️ Security Notes

- **Change the secret key** before deploying to production
- User passwords are securely hashed using Werkzeug
- File uploads are sanitized with `secure_filename()`
- User data is isolated per user account

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author

- **M.Chandran** - Initial work

---

⭐ If you found this project useful, please give it a star!
