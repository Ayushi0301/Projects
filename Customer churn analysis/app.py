from flask import Flask, render_template, request
from flask import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
model = pickle.load(open('SVM.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

num_data = pd.read_csv("E:/End to end project/Telecom Churn Prediction/num_data.csv")
scaler = MinMaxScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        tenure = int(request.form['tenure'])        
        MonthlyCharges=float(request.form['MonthlyCharges'])        
        TotalCharges=float(request.form['TotalCharges'])
        
        temp = pd.DataFrame(index=[1])
        temp['tenure']= tenure
        temp['MonthlyCharges'] = MonthlyCharges
        temp['TotalCharges'] = TotalCharges
        
        scaler.fit(num_data)
        scaled = temp.copy()
        scaled = scaler.transform(scaled)
        
        tenure = scaled[0][0]
        MonthlyCharges = scaled[0][1]
        TotalCharges = scaled[0][2]
        
        SeniorCitizen = request.form['SeniorCitizen']
        if(SeniorCitizen=='Yes'):
            SeniorCitizen=1                
        else:
            SeniorCitizen=0
        Partner = request.form['Partner']
        if(Partner=='Yes'):
            Partner=1                
        else:
            Partner=0
        Dependents = request.form['Dependents']
        if(Dependents=='Yes'):
            Dependents=1                
        else:
            Dependents=0                 
        InternetService = request.form['InternetService']
        if(InternetService=='No'):
            InternetService_Fiber_optic = 0
            InternetService_No = 1
        elif(InternetService=='Fiber optic'):
            InternetService_Fiber_optic = 1
            InternetService_No = 0
        else:
            InternetService_Fiber_optic = 0
            InternetService_No = 0
        OnlineSecurity = request.form['OnlineSecurity']
        if(OnlineSecurity=='Yes'):
            OnlineSecurity=1                
        else:
            OnlineSecurity=0    
        OnlineBackup = request.form['OnlineBackup']
        if(OnlineBackup=='Yes'):
            OnlineBackup=1                
        else:
            OnlineBackup=0     
        DeviceProtection = request.form['DeviceProtection']
        if(DeviceProtection=='Yes'):
            DeviceProtection=1                
        else:
            DeviceProtection=0    
        TechSupport = request.form['TechSupport']
        if(TechSupport=='Yes'):
            TechSupport=1                
        else:
            TechSupport=0    
        StreamingTV = request.form['StreamingTV']
        if(StreamingTV=='Yes'):
            StreamingTV=1                
        else:
            StreamingTV=0
        StreamingMovies = request.form['StreamingMovies']
        if(StreamingMovies=='Yes'):
            StreamingMovies=1                
        else:
            StreamingMovies=0
        Contract = request.form['Contract']
        if(Contract=='One year'):
            Contract_One_year = 1
            Contract_Two_year = 0
        elif(Contract=='Two year'):
            Contract_One_year = 0
            Contract_Two_year = 1
        else:
            Contract_One_year = 0
            Contract_Two_year = 0
        PaperlessBilling = request.form['PaperlessBilling']
        if(PaperlessBilling=='Yes'):
            PaperlessBilling=1                
        else:
            PaperlessBilling=0
        PaymentMethod = request.form['PaymentMethod']
        if(PaymentMethod=='Electronic check'):
            PaymentMethod_Electronic_check=1
            PaymentMethod_Mailed_check = 0
            PaymentMethod_Credit_card_automatic = 0
        elif(PaymentMethod=='Mailed check'):
            PaymentMethod_Electronic_check=0
            PaymentMethod_Mailed_check = 1
            PaymentMethod_Credit_card_automatic = 0
        elif(PaymentMethod=='Credit_card (automatic)'):
            PaymentMethod_Electronic_check=0
            PaymentMethod_Mailed_check = 0
            PaymentMethod_Credit_card_automatic = 1
        else:
            PaymentMethod_Electronic_check=0
            PaymentMethod_Mailed_check = 0
            PaymentMethod_Credit_card_automatic = 0    
            
        prediction=model.predict([[tenure, MonthlyCharges, TotalCharges, SeniorCitizen,Partner, Dependents,
       OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
       StreamingTV, StreamingMovies, PaperlessBilling,InternetService_Fiber_optic,
       InternetService_No, Contract_One_year,
       Contract_Two_year,
       PaymentMethod_Credit_card_automatic,
       PaymentMethod_Electronic_check, PaymentMethod_Mailed_check]])
    
        probability=model.predict_proba([[tenure, MonthlyCharges, TotalCharges, SeniorCitizen,Partner, Dependents,
       OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
       StreamingTV, StreamingMovies, PaperlessBilling,InternetService_Fiber_optic,
       InternetService_No, Contract_One_year,
       Contract_Two_year,
       PaymentMethod_Credit_card_automatic,
       PaymentMethod_Electronic_check, PaymentMethod_Mailed_check]])[:,1][0]
        probability = round(probability,2)*100
        if prediction == 1:
            return render_template('index.html',prediction_text="Custmer can churn. Probability of churn is {} %".format(probability))
            
        else:
            
            return render_template('index.html',prediction_text="Custmer will not churn. Probability of churn is {} %".format(probability))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

