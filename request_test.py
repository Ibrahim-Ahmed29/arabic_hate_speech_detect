import requests

# URL of your Flask API endpoint
url = 'https://ibrahimahmed.pythonanywhere.com/predict'  # Replace with your actual URL

# Sample data for prediction
data = {
    'text': 'i am organization'
}

# Send POST request to the API endpoint
response = requests.post(url, json=data)

# Check if request was successful
if response.status_code == 200:
    # Print the prediction result
    print(response.json())
else:
    print('Error:', response.status_code)
