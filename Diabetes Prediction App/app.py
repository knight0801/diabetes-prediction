from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Models
scaler = pickle.load(open('scaler.pkl', 'rb'))
lr = pickle.load(open('lr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get form data
        Age = int(request.form['Age'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        SkinThickness = int(request.form['SkinThickness'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])

        # Create array for prediction
        input_data = np.array([[Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        scaled_data = scaler.transform(input_data)
        prediction = int(lr.predict(scaled_data)[0])

        if prediction == 1:
            result_text = "The patient is likely to have diabetes."
        else:
            result_text = "The patient is unlikely to have diabetes."

        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
