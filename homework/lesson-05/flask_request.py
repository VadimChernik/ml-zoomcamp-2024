import requests

# Here is your API url:
url = 'http://localhost:9696/predict'

# Here are your customer characteristics, in a dict:
customer_id = 'xyz-123'
client = {"job": "student", "duration": 280, "poutcome": "failure"}

# Here your get your response from your API using the customer above
response = requests.post(url, json=client).json()
print(response)

# Hereyou precise if a promotional email should be sent to this customer
if response['churn'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)