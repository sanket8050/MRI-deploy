from flask import Flask, request, render_template
import pickle
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from werkzeug.utils import secure_filename

# Load model
with open('model/brain_tumor_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
import os

# Ensure the upload directory exists
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # ⚠️ Update based on your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # CNN expects batch
    img_array /= 255.0  # normalize if needed
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No selected file!")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        labels = ['glioma', 'meningioma','notumor','pitutary']  # customize based on your model
        result = labels[predicted_class]

        return render_template('index.html',
                               prediction_text=f'Prediction: {result}',
                               image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
