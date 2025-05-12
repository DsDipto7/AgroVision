from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import requests
import json

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('my_model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Constants
IMAGE_SIZE = 255
GEMINI_API_KEY = "AIzaSyAeZxw6NfGTKJ79RfElmSo6Nk9ubCwS29M"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Prediction Function
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Home Route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            predicted_class, confidence = predict(img)
            return render_template('index.html', image_path=filepath,
                                   actual_label=predicted_class,
                                   predicted_label=predicted_class,
                                   confidence=confidence)
    return render_template('index.html', message='Upload an image')

# Allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Gemini Chatbot Route
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": user_message}]
        }]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        gemini_reply = response.json()['candidates'][0]['content']['parts'][0]['text']
        return jsonify({'reply': gemini_reply})
    else:
        return jsonify({'reply': 'Sorry, something went wrong with Gemini API.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
