import pickle
from flask import Flask, request, app, jsonify, render_template, url_for
import pandas as pd
import numpy as np

app=Flask(__name__)

 ## Load the model
rf_model=pickle.load(open('rf_petrol_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())))
    new_data=np.array(list(data.values()))
    output=rf_model.predict([new_data])
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[int(x) for x in request.form.values()]
    final_data=np.array(list(data))
    print(final_data)
    output=rf_model.predict([final_data])
    output=output[0]
    return render_template("home.html",prediction_text="Petrol Price Prediction is {}".format(output))



if __name__=='__main__':
    app.run(debug=True)
