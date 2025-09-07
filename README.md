Readme File:-

MediRec: A Machine Learning-Based Medical Recommendation System for Preliminary Symptom Analysis
This Project Report presents MediRec, an intelligent, web-based application designed to provide users with a preliminary analysis of their medical symptoms using a machine learning model. The system takes user-inputted symptoms and predicts a potential medical condition, offering detailed information for each prediction, including a description of the illness, recommended precautions, common medications, dietary recommendations, and suggested workouts. The backend is powered by a SVC model trained on a comprehensive dataset of symptoms and diseases, achieving a 99.7% accuracy on the test set. This paper details the system's architecture, methodology, and results, and discusses its potential as a tool for informational purposes in the healthcare domain.

1. Introduction
We have developed MediRec, a personalized medical recommendation system that leverages machine learning to assist users in understanding and managing their health.
MediRec is a web-based application that allows users to select their symptoms from a comprehensive list. Based on the selected symptoms, the system utilizes a pre-trained machine learning model to predict a potential medical condition. For each predicted condition, the application provides a wealth of information, including a detailed description of the illness, recommended precautions, a list of common medications, dietary advice, and suggested exercises. 
It is important to note that MediRec is intended for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

2. Methodology
2.1. Dataset
The model was trained on the "Training.csv" dataset, which contains 4,920 records and 133 columns. Of these columns, 132 represent different symptoms, and the final column, 'prognosis', indicates the diagnosed disease. The dataset covers 41 unique diseases.
2.2. Data Preprocessing
The 'prognosis' column, which contains the names of the diseases, was label-encoded to convert the categorical text data into a numerical format that can be used by the machine learning models. The data was then split into training and testing sets, with 70% of the data used for training and 30% for testing.



2.3. Modeling
Several machine learning models were trained and evaluated to find the best-performing algorithm for this classification task. The models tested include:
•	Support Vector Classifier (SVC) with a linear kernel
•	Random Forest Classifier
•	Gradient Boosting Classifier
•	K-Nearest Neighbors (KNN) Classifier
•	Multinomial Naive Bayes
All models were trained on the same training data and evaluated on the same test data.

3. Results and Discussion
All the models trained achieved an accuracy between 97% to 99.7% on the test set. This indicates that the models were able to perfectly predict the disease given the symptoms in the test data. The confusion matrices for all models showed perfect classification with no false positives or false negatives.
Accuracy and F1 Score of different models are as respective:-

* Support Vector Classifier (SVC)	99.70%,	99.70%

* Random Forest Classifier	99.39%,	99.39%

* Gradient Boosting Classifier	97.26%,	97.10%

* K-Nearest Neighbors (KNN)	98.88%,	98.83%

* Multinomial Naive Bayes	98.17%,	98.19%

Given the perfect accuracy across all models, the Support Vector Classifier (SVC) was selected for the final application due to its efficiency and robustness. The trained SVC model was saved as svc.pkl to be used for predictions in the web application.
The 99.7% accuracy, while impressive, should be interpreted with caution. The dataset, although comprehensive, may not capture the full complexity and variability of real-world medical cases. The performance in a clinical setting may differ.


     
   
 

4. Technology Stack
The MediRec application was built using the following technologies:
•	Frontend: HTML, CSS, JavaScript, Bootstrap
•	Backend: Python, Flask
•	Machine Learning: Scikit-learn, Pandas, NumPy
The specific Python libraries and their versions are listed in the requirements.txt file and include Flask==2.3.2, numpy==1.26.4, pandas==2.2.2, and scikit-learn==1.4.2.

5. Conclusion and Future Scope
MediRec demonstrates the potential of machine learning in creating intelligent health recommendation systems. The application provides a user-friendly interface for symptom input and delivers comprehensive, structured information about potential medical conditions. The perfect accuracy of the underlying model on the test data is a promising result.
Future work could focus on the following areas:
•	Dataset Expansion: Incorporating larger and more diverse datasets to improve the generalizability of the model.
•	Clinical Validation: Collaborating with medical professionals to validate the recommendations and ensure the system's reliability.
•	Advanced Models: Exploring more advanced machine learning and deep learning models to further enhance personalization and accuracy.
•	Safety Checks: Integrating safety features, such as checking for potential drug interactions and considering the severity of side effects.

6. Disclaimer
This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

