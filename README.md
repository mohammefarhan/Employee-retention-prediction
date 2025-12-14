🚀 Employee-Retention-Prediction
<p align="center"> <b>Predict employee attrition using Machine Learning</b><br> Hyperparameter-tuned XGBoost · Streamlit Web App · End-to-End ML Project </p>
📌 Overview

Employee attrition directly impacts productivity, hiring cost, and team stability.
This project predicts whether an employee is likely to leave or stay based on historical and behavioral data using a tuned XGBoost model.

The solution covers the complete ML lifecycle:

Data preprocessing → Model training & tuning → Evaluation → Deployment

🧠 Why XGBoost?

Captures non-linear relationships

Handles categorical-heavy tabular data

Strong performance with limited feature engineering

Excellent balance of precision & recall

Hyperparameters are optimized using RandomizedSearchCV with F1-score as the primary metric.

✨ Key Features

✅ Predict employee retention risk
✅ Probability-based output (confidence score)
✅ Hyperparameter-tuned model
✅ Clean & interactive Streamlit UI
✅ Production-style deployment workflow

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

Identifiers and high-cardinality location features were removed to improve model robustness and deployment stability.

📊 Model Evaluation

The model is evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

The tuned XGBoost model outperforms baseline models by learning complex interactions between employee attributes.

🌐 Web Application (Streamlit)

The Streamlit app allows users to:

Enter employee details through a clean UI

Get instant predictions (Stay / Leave)

View confidence scores

Demonstrate real-world ML deployment

📁 Project Structure
Employee-retention-prediction/
│
├── app.py                   # Streamlit web application
├── train_xgboost.py         # Model training & tuning
├── xgboost_fraud_model.pkl  # Trained XGBoost model
├── label_encoders.pkl       # Encoders for categorical features
├── aug_train.csv            # Training dataset
├── requirements.txt         # Dependencies
└── README.md

⚙️ Installation & Usage
1️⃣ Clone the repository
git clone https://github.com/your-username/Employee-retention-prediction.git
cd Employee-retention-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model (optional)
python train_xgboost.py

4️⃣ Run the web app
streamlit run app.py

📈 Use Cases

HR analytics & workforce planning

Employee churn risk identification

Data-driven retention strategies

Portfolio / academic ML project

🏁 Key Takeaway

This project demonstrates how machine learning can be applied end-to-end to solve a real business problem — from data preprocessing and model tuning to deployment in an interactive web application.

👤 Author

Farhan
Machine Learning & Data Science
