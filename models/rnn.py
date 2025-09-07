# Title: Personalized Medical Recommendation System with Machine Learning 

# Load dataset & tools 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pickle

# Load dataset
dataset = pd.read_csv('datasets/Training.csv')

# Separate features and target
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# Encode target labels
le = LabelEncoder()
Y = le.fit_transform(y)

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Add Noise
def add_noise(X, noise_level=0.3):   
    X_noisy = X.copy()
    noise = np.random.normal(0, noise_level, X_noisy.shape)
    return X_noisy + noise

X_train_noisy = add_noise(X_train, 0.3)
X_test_noisy = add_noise(X_test, 0.3)

#  Train & Evaluate (SVC) 
svc = SVC(kernel='linear', C=0.8)
svc.fit(X_train_noisy, y_train)

predictions = svc.predict(X_test_noisy)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
cm = confusion_matrix(y_test, predictions)

print(f"💻 SVC Accuracy: {acc:.4f}")
print(f"💻 SVC F1 Score: {f1:.4f}")
print(f"💻 SVC Confusion Matrix:\n{cm}")

# Save Final Model 
pickle.dump(svc, open('models/svc.pkl', 'wb'))
print("✅ Model saved as svc.pkl")

# Load model for predictions
svc = pickle.load(open('models/svc.pkl', 'rb'))
print("Final SVC test accuracy:", accuracy_score(y_test, svc.predict(X_test_noisy)))

# Recommendation System Data
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [m for m in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [d for d in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Prediction Function 
symptoms_dict = {name: idx for idx, name in enumerate(X.columns)}
diseases_list = dict(enumerate(le.classes_))

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Auto Test Predictions 
test_cases = [
    ["itching", "skin_rash", "nodal_skin_eruptions"],
    ["yellow_crust_ooze", "red_sore_around_nose", "blister"]
]

for case in test_cases:
    predicted_disease = get_predicted_value(case)
    desc, pre, med, die, wrkout = helper(predicted_disease)

    print("\n  Test Case  ")
    print(f"Symptoms: {case}")
    print("  Predicted Disease ")
    print(predicted_disease)
    print("  Description  =")
    print(desc)
    print("  Precautions  =")
    for i, p_i in enumerate(pre[0], start=1):
        print(i, ": ", p_i)
    print("  Medications  =")
    for i, m_i in enumerate(med, start=1):
        print(i, ": ", m_i)
    print("  Workout  =")
    for i, w_i in enumerate(wrkout, start=1):
        print(i, ": ", w_i)
    print("  Diets  =")
    for i, d_i in enumerate(die, start=1):
        print(i, ": ", d_i)

# sklearn version
import sklearn
print("sklearn version:", sklearn.__version__)
