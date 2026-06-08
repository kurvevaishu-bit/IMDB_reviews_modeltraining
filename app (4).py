
import pickle
from flask import Flask, request, jsonify

# Load the pre-trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return "Sentiment Analysis API. Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data['review']

    # Vectorize the input review
    review_vectorized = vectorizer.transform([review])

    # Make prediction
    prediction = model.predict(review_vectorized)
    
    return jsonify({
        'sentiment': prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
