📊 Employee-Retention-Prediction

A machine learning–based web application that predicts whether an employee is likely to leave an organization. The system uses a hyperparameter-tuned XGBoost model trained on structured employee data and is deployed using Streamlit for real-time predictions.

🚀 Project Overview

Employee attrition is costly and difficult to manage proactively. This project aims to help organizations identify high-risk employees early by analyzing factors such as experience, education, company details, and training history.

The model learns complex, non-linear patterns from historical data and provides both:

A binary prediction (Likely to Leave / Likely to Stay)

A probability score indicating confidence

🧠 Machine Learning Approach

Algorithm: XGBoost (Gradient Boosting)

Optimization Metric: F1-score

Hyperparameter Tuning: RandomizedSearchCV

Problem Type: Binary Classification

Why XGBoost:

Handles non-linear relationships

Works well with categorical-heavy tabular data

Strong generalization performance

🗂️ Features Used

Gender

Relevant Experience

University Enrollment

Education Level

Major Discipline

Years of Experience

Company Size

Company Type

Years Since Last Job Change

Training Hours

(Identifier and high-cardinality location features were intentionally removed to improve robustness and deployment stability.)

🧪 Model Evaluation

The model is evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Validation results show that the tuned XGBoost model outperforms baseline models by effectively balancing precision and recall.

🌐 Web Application (Streamlit)

The Streamlit app allows users to:

Enter employee details via an intuitive UI

Get instant retention predictions

View confidence scores

Demonstrate real-world deployment of an ML model

📁 Project Structure
Employee-retention-prediction/
│
├── app.py                     # Streamlit application
├── train_xgboost.py           # Model training & tuning script
├── xgboost_fraud_model.pkl    # Trained XGBoost model
├── label_encoders.pkl         # Saved encoders for categorical features
├── aug_train.csv              # Training dataset
├── requirements.txt           # Project dependencies
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/Employee-retention-prediction.git
cd Employee-retention-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model (optional)
python train_xgboost.py

4️⃣ Run the Streamlit app
streamlit run app.py

📌 Use Cases

HR analytics and workforce planning

Early identification of attrition risk

Data-driven employee retention strategies

Academic and portfolio demonstration project

📄 Key Takeaway

This project demonstrates a complete end-to-end machine learning workflow — from data preprocessing and model tuning to deployment in a user-friendly web application.

👤 Author

Farhan
Machine Learning & Data Science Enthusiast
