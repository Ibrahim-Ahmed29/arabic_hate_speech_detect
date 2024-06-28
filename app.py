from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model pipeline
model = joblib.load('model_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    prediction = model.predict([text])
    response = {
        'prediction': int(prediction[0]),
        'message': 'Hate Speech' if prediction[0] == 1 else 'Not Hate Speech'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
