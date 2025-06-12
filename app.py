from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('lr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        data = [
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age']),
        ]
        input_data = np.array([data])
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]

        result = "The patient is likely to have diabetes." if prediction == 1 else "The patient is unlikely to have diabetes."
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
