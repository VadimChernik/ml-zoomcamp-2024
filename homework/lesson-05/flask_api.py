import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from flask import Flask
from flask import request
from flask import jsonify

# Define your app name here
app=Flask('churn')

# Load the model and dv
model_file = "model1.bin"
dv_file = "dv.bin"
with open(model_file, 'rb') as model:
    model = pickle.load(model)
with open(dv_file, 'rb') as dv:
    dv = pickle.load(dv)

# Use this decorator: this will be the predict that gives you the output of a request
@app.route('/predict', methods=['POST'])
def predict():
    # Get your customer informations
    customer= request.get_json()
    
    # Make your prediction using the loaded model and dv
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    
    # Obtain the result: you have to specify the type of the variables
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    # Return your result in a json format
    return jsonify(result)

# Here, when you run your script with python in dev environment, you can debug it
if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)