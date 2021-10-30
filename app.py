from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))


def calc(data):
    #separate data
    prg = data['Pregnancies']
    glu = data['Glucose']
    bp = data['BloodPressure']
    st = data['SkinThickness']
    ins = data['Insulin']
    bmi = data['BMI']
    dpf = data['DiabetesPedigreeFunction']
    age = data['Age']
    #convert to numpy array
    db_data = (np.array([int(prg), int(glu), int(bp), int(st), int(ins), int(bmi), int(dpf), int(age)]))
    db_data = db_data.reshape(1, -1)
    print(db_data.shape)
    #calculate
    return model.predict(db_data)

@app.route('/')
def index():
    return render_template('index.html')

#When the user clicks the button, the server will send a POST request to the url /predict
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        print("Got post")
        #Get the data from the form
        data = request.form.to_dict()
        print(data)
        #Convert the data to a dictionary
        res = calc(data)
        data = {k:v for k, v in data.items()}
        print(data)
        #Get the prediction from the model
        prediction = calc(data)
        #Return the prediction
        if prediction[0] == 1:
            return render_template('index.html', probability="You are at risk of diabetes")
        else:
            return render_template('index.html', probability="You are not at risk of diabetes")

if __name__ == '__main__':
    app.run(debug=True)