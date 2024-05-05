from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_url_path='/static')

df1=pd.read_excel('Appendicitis1.xlsx')

# Assume 'X' contains your features and 'y' contains your target variable
X = df1.drop(columns=['Management', 'Severity', 'Diagnosis','BMI','Length_of_Stay','Diagnosis_Presumptive','Alvarado_Score','Paedriatic_Appendicitis_Score'])  # Features
y = df1['Diagnosis']  # Target variable

# Convert categorical variables to numerical using one-hot encoding or label encoding
# Make sure all features are categorical for chi-square test
X = pd.get_dummies(X)  # One-hot encoding for simplicity, replace with appropriate encoding method

# Select top k features using chi-square test
k = 10  # Number of features to select
chi2_selector = SelectKBest(chi2, k=k)
X_kbest = chi2_selector.fit_transform(X, y)

# Get the indices of the selected features
selected_indices = chi2_selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X.columns[selected_indices]

# Print the names of the selected features
print("Selected features:")
print(selected_features)
# Create a new DataFrame with selected features
df2 = df1.loc[:, selected_features].copy()

# Add 'Age' and 'Height' columns to the selected DataFrame
df2['Lower_Right_Abd_Pain'] = df1['Lower_Right_Abd_Pain']
df2['Body_Temperature'] = df1['Body_Temperature']
df2['Age'] = df1['Age']
df2['Height'] = df1['Height']
df2['Diagnosis'] = df1['Diagnosis']

# Display the first few rows of the new DataFrame
print(df2.head())

# Assume 'X' contains your features and 'y' contains your target variable
X = df2.drop(columns=['Diagnosis'])  # Features
y = df2['Diagnosis']  # Target variable

# Count the number of samples in each class
non_appendicitis_count = np.sum(y == 1)
appendicitis_count = np.sum(y == 0)

# Desired number of rows
desired_rows = 1100

# Calculate the number of synthetic samples needed for each class
non_appendicitis_synthetic = max(desired_rows - non_appendicitis_count, 0)
appendicitis_synthetic = max(desired_rows - appendicitis_count, 0)

# Apply SMOTE with the adjusted sampling strategy
smote = SMOTE(sampling_strategy={1: non_appendicitis_synthetic, 0: appendicitis_synthetic}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the shape of the resampled dataset
print("Shape of X_resampled:", X_resampled.shape)
print("Shape of y_resampled:", y_resampled.shape)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': len(y.unique()),  # Number of classes
    'metric': 'multi_error'  # Error rate for multiclass classification
}

# Convert dataset to LightGBM format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Train the LightGBM model
model = lgb.train(params, train_set=train_data, num_boost_round=100, valid_sets=[test_data])

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    age = float(request.form['age'])
    sex = 1 if request.form['sex'].lower() == 'male' else 0
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    body_temperature = float(request.form['body_temperature'])
    migratory_pain = 1 if request.form['migratory_pain'].lower() == 'yes' else 0
    lower_right_abd_pain = 1 if request.form['lower_right_abd_pain'].lower() == 'yes' else 0
    contralateral_rebound_tenderness = 1 if request.form['contralateral_rebound_tenderness'].lower() == 'yes' else 0
    nausea = 1 if request.form['nausea'].lower() == 'yes' else 0
    Loss_of_Appetite = 1 if request.form['Loss_of_Appetite'].lower() == 'yes' else 0
    WBC_Count = float(request.form['WBC_Count'])
    Neutrophil_Percentage = float(request.form['Neutrophil_Percentage'])
    Neutrophilia = 1 if request.form['Neutrophilia'].lower() == 'yes' else 0
    CRP = float(request.form['CRP'])

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[age, sex, height, weight, body_temperature, migratory_pain,
                                lower_right_abd_pain, contralateral_rebound_tenderness,
                                Loss_of_Appetite, nausea, WBC_Count, Neutrophil_Percentage, Neutrophilia, CRP]],
                              columns=X.columns)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)  # Get the index of the maximum value along the row axis

    # Print y_test and y_pred_class
    print("Actual values (y_test):", y_test)
    print("Predicted values (y_pred_class):", y_pred_class)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_class)
    print("Accuracy:", accuracy)

    # Make predictions
    pred = model.predict(input_data)
    pred_class = np.argmax(pred, axis=1)

    # Determine the diagnosis based on the predicted class
    diagnosis = "Appendicitis" if pred_class[0] == 0 else "No Appendicitis"

    # Return the diagnosis as a response
    return diagnosis

@app.route('/reset', methods=['POST'])
def clear():
    # Clear the diagnosis result
    diagnosis_result = " "
    # Return an empty response to clear the form fields
    return jsonify({'diagnosis_result': diagnosis_result})

if __name__ == '__main__':
    app.run(debug=False)