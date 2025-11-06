# Employee Attrition Analysis System

A comprehensive machine learning-powered web application for analyzing and predicting employee attrition. This system helps HR departments identify at-risk employees and understand the key factors contributing to employee turnover.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Components](#project-components)
- [Database Schema](#database-schema)
- [Machine Learning Models](#machine-learning-models)
- [Web Application](#web-application)
- [Data Analysis](#data-analysis)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Employee Attrition Analysis System is a full-stack application that combines machine learning models with an interactive web interface to help organizations:

- **Predict** employee attrition risk based on various employee attributes
- **Analyze** patterns and trends in employee data
- **Visualize** key insights through interactive dashboards
- **Track** prediction history and employee records
- **Identify** critical factors that influence employee retention

The system uses multiple machine learning algorithms (Logistic Regression, Random Forest, Decision Tree) to predict attrition and provides a user-friendly Streamlit-based web interface for real-time predictions and data analysis.

## ✨ Features

### Core Features

1. **User Authentication**
   - Secure user registration and login system
   - Password hashing using bcrypt
   - User-specific data isolation

2. **Employee Data Management**
   - Store and manage employee information
   - Track employee attributes (age, salary, job satisfaction, etc.)
   - Automatic data persistence in SQLite database

3. **Attrition Prediction**
   - Real-time risk assessment for individual employees
   - Multi-factor risk scoring algorithm
   - Visual risk indicators (gauge charts, probability scores)
   - Detailed factor contribution analysis

4. **Data Analysis Dashboard**
   - Interactive visualizations using Plotly
   - Department-wise attrition analysis
   - Age distribution analysis
   - Salary vs. tenure correlations
   - Job satisfaction and work-life balance insights

5. **Prediction History**
   - Track all predictions made by users
   - View historical data in table and chart formats
   - Department-wise risk analysis
   - Timeline visualization of predictions
   - Export functionality (CSV download)

6. **Machine Learning Pipeline**
   - Automated data preprocessing
   - Multiple model training and evaluation
   - Feature importance analysis
   - Model performance metrics

## 📁 Project Structure

```
Employee-Attrition-System/
├── app.py                      # Main Streamlit web application
├── main.py                     # Standalone analysis script
├── database.py                 # Database models and operations
├── requirements.txt            # Python dependencies
├── package.json                # Frontend dependencies (Material-UI)
├── employee_attrition.db       # SQLite database file
│
├── src/
│   ├── main.py                 # Modular main script
│   │
│   ├── data/
│   │   ├── preprocessing.py   # Data loading and preprocessing functions
│   │   └── raw/
│   │       └── data/
│   │           └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
│   │
│   ├── models/
│   │   └── train.py            # ML model training and evaluation
│   │
│   ├── visualization/
│   │   └── plots.py            # Data visualization functions
│   │
│   └── utils/                  # Utility functions
│
├── Generated Visualizations/
│   ├── attrition_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── marital_status_vs_attrition.png
│
└── README.md                   # Project documentation
```

## 🛠 Technologies Used

### Backend & Machine Learning
- **Python 3.x** - Core programming language
- **Streamlit** - Web application framework
- **SQLAlchemy** - ORM for database operations
- **SQLite** - Lightweight database
- **scikit-learn** - Machine learning library
  - Logistic Regression
  - Random Forest Classifier
  - Decision Tree Classifier
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **bcrypt** - Password hashing

### Data Visualization
- **Plotly** - Interactive web-based visualizations
- **Matplotlib** - Static plot generation
- **Seaborn** - Statistical data visualization

### Frontend
- **Material-UI (MUI)** - React component library (for potential frontend extensions)

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Employee-Attrition-System
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install additional Streamlit dependencies (if not included)
pip install streamlit plotly bcrypt sqlalchemy
```

### Step 4: Initialize Database

The database will be automatically created when you first run the application. The SQLite database file (`employee_attrition.db`) will be generated in the project root directory.

### Step 5: Run the Application

```bash
# Start the Streamlit web application
streamlit run app.py

# Or run the standalone analysis script
python main.py

# Or run the modular main script
python src/main.py
```

The web application will be available at `http://localhost:8501`

## 📖 Usage

### Web Application (Streamlit)

1. **Registration/Login**
   - Create a new account or login with existing credentials
   - Each user has isolated data and prediction history

2. **Home Dashboard**
   - Overview of the application features
   - Quick navigation to different sections

3. **Data Analysis**
   - View interactive visualizations:
     - Attrition by department
     - Age distribution
     - Salary vs. years at company
     - Job satisfaction vs. work-life balance

4. **Attrition Prediction**
   - Enter employee information:
     - Basic details (name, age, gender)
     - Job information (role, department, level)
     - Satisfaction metrics (job, environment, work-life balance)
     - Performance indicators
   - Click "Predict Attrition Risk" to get:
     - Risk probability percentage
     - Visual gauge chart
     - Factor contribution analysis

5. **Prediction History**
   - View all past predictions
   - Analyze trends and patterns
   - Export data as CSV

### Standalone Analysis Script

Run the analysis script to:
- Load and preprocess the dataset
- Train machine learning models
- Generate visualizations
- Evaluate model performance

```bash
python main.py
# or
python src/main.py
```

## 🔧 Project Components

### 1. Database Module (`database.py`)

Manages all database operations using SQLAlchemy ORM:

- **User Model**: Stores user credentials and account information
- **Employee Model**: Stores employee data and attributes
- **PredictionHistory Model**: Tracks all prediction records

Key Functions:
- `create_user()` - Register new users
- `verify_user()` - Authenticate users
- `save_employee_data()` - Store employee records
- `get_employee_data()` - Retrieve employee data
- `save_prediction_history()` - Log predictions
- `get_prediction_history()` - Retrieve prediction history

### 2. Data Preprocessing (`src/data/preprocessing.py`)

Handles data preparation:

- **load_data()**: Loads CSV data and performs initial cleaning
- **preprocess_data()**: 
  - One-hot encoding for categorical variables
  - Standard scaling for numerical features
  - Unit conversions (USD to INR, miles to km)

### 3. Machine Learning Models (`src/models/train.py`)

Implements model training and evaluation:

- **prepare_train_test_data()**: Splits data into training and testing sets
- **train_logistic_regression()**: Trains logistic regression model
- **train_random_forest()**: Trains random forest classifier
- **evaluate_model()**: Calculates accuracy, classification report, and ROC-AUC
- **get_feature_importance()**: Extracts feature importance from models

### 4. Visualization Module (`src/visualization/plots.py`)

Creates various data visualizations:

- Marital status vs. attrition
- Department vs. attrition
- Correlation heatmap
- Feature importance charts
- Gender vs. attrition
- Age distribution analysis

### 5. Web Application (`app.py`)

Streamlit-based interactive web interface with:

- **Authentication System**: Login/registration with secure password hashing
- **Multi-page Navigation**: Home, Data Analysis, Prediction, History
- **Interactive Dashboards**: Real-time data visualization
- **Prediction Interface**: User-friendly form for employee data input
- **Risk Scoring Algorithm**: Multi-factor risk assessment

## 🗄️ Database Schema

### Users Table
- `id` (Primary Key)
- `username` (Unique)
- `password_hash` (Encrypted)
- `email` (Unique)
- `created_at` (Timestamp)

### Employees Table
- `id` (Primary Key)
- `employee_name`
- `age`
- `gender`
- `relationship_status`
- `distance_from_home`
- `monthly_income`
- `job_role`
- `department`
- `years_at_company`
- `work_life_balance`
- `job_level`
- `training_times_last_year`
- `job_satisfaction`
- `performance_rating`
- `environment_satisfaction`
- `attrition` (Boolean)
- `user_id` (Foreign Key)
- `created_at` (Timestamp)

### Prediction History Table
- `id` (Primary Key)
- All employee fields (same as Employees table)
- `attrition_risk` (Float: 0-1 probability)
- `prediction_date` (Timestamp)
- `user_id` (Foreign Key)

## 🤖 Machine Learning Models

### Model Comparison

The system trains and evaluates three machine learning models:

1. **Logistic Regression**
   - Linear classification model
   - Balanced class weights for handling imbalanced data
   - Fast training and prediction

2. **Random Forest Classifier**
   - Ensemble method using multiple decision trees
   - Provides feature importance scores
   - Handles non-linear relationships

3. **Decision Tree Classifier**
   - Single tree-based model
   - Maximum depth: 5 (to prevent overfitting)
   - Easy to interpret

### Evaluation Metrics

- **Accuracy**: Overall prediction correctness
- **Classification Report**: Precision, recall, F1-score per class
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: Visual representation of predictions

### Feature Importance

The Random Forest model identifies the top contributing factors:
- Monthly Income
- Age
- Years at Company
- Job Satisfaction
- Work-Life Balance
- And more...

## 🌐 Web Application

### Risk Scoring Algorithm

The prediction system uses a weighted scoring approach considering:

1. **Age Factor** (0.05-0.18)
   - Younger employees (<30) have higher risk
   - Older employees (>50) have lower risk

2. **Income Factor** (0.05-0.22)
   - Lower income increases attrition risk
   - Higher income reduces risk

3. **Tenure Factor** (0.05-0.17)
   - Newer employees (<2 years) have higher risk
   - Longer tenure reduces risk

4. **Work-Life Balance** (0.02-0.20)
   - Poor balance significantly increases risk
   - Excellent balance reduces risk

5. **Job Satisfaction** (0.02-0.22)
   - Low satisfaction is a major risk factor
   - High satisfaction indicates retention

6. **Environment Satisfaction** (0.02-0.15)
   - Work environment quality affects retention

7. **Performance Rating** (0.02-0.15)
   - Lower ratings correlate with higher risk

8. **Distance from Home** (0.03-0.12)
   - Longer commutes increase risk

9. **Job Level** (0.03-0.10)
   - Entry-level positions have higher turnover

The final risk score is normalized between 0.15 and 0.9, with scores above 0.5 indicating high attrition risk.

## 📊 Data Analysis

### Generated Visualizations

The system produces several static visualizations:

1. **Attrition Distribution**: Overall distribution of attrition cases
2. **Correlation Heatmap**: Feature correlations in the dataset
3. **Feature Importance**: Top 5 most important features
4. **Marital Status vs. Attrition**: Relationship between marital status and turnover
5. **Department vs. Attrition**: Department-wise attrition patterns

### Interactive Visualizations (Web App)

- Real-time charts using Plotly
- Interactive filters and hover information
- Responsive design for different screen sizes

## 🔒 Security Features

- **Password Hashing**: Uses bcrypt for secure password storage
- **User Isolation**: Each user can only access their own data
- **SQL Injection Prevention**: Uses parameterized queries via SQLAlchemy
- **Session Management**: Streamlit session state for user authentication

## 📝 Notes

- The dataset is converted to Indian metrics (INR for currency, km for distance)
- The system uses class balancing techniques to handle imbalanced datasets
- All visualizations are saved as PNG files in the project root
- The database is automatically initialized on first run

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available for educational and commercial use.

---

**Developed with ❤️ for better employee retention and HR analytics**
