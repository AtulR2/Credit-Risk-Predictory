from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model_pred.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        status = int(request.form['status'])
        credhist = int(request.form['credit_history'])
        purpose = int(request.form['purpose'])
        savings = int(request.form['savings'])
        emp = int(request.form['employment_duration'])
        insrate = int(request.form['installment_rate'])
        personal = int(request.form['personal_status_sex'])
        othdebt = int(request.form['other_debtors'])
        residence = int(request.form['present_residence'])
        prpty = int(request.form['property'])
        othinsplan = int(request.form['other_installment_plans'])
        housing = int(request.form['housing'])
        numcred = int(request.form['number_credits'])
        job = int(request.form['job'])
        ppllib = int(request.form['people_liable'])
        forwork = int(request.form['foreign_worker'])
        logamt = int(request.form['log_amount'])
        logage = int(request.form['log_age'])
        logdur = int(request.form['log_duration'])
        
        
        
        data = np.array([[status,credhist,purpose,savings,emp,insrate,personal,othdebt,residence,prpty,othinsplan,housing,numcred,job,ppllib,forwork,logamt,logage,logdur]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
