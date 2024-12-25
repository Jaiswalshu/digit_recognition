from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_images/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = load_model('Digit_Recognition.keras')

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", result=None)

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Preprocess the image
        image = load_img(filepath, target_size=(28, 28), color_mode="grayscale")  # Convert to grayscale
        image_array = img_to_array(image)  # Convert to NumPy array
        image_array = image_array / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 28, 28, 1)

        # Make a prediction
        predictions = model.predict(image_array)
        predicted_digit = np.argmax(predictions[0])

        return render_template("index.html", result=predicted_digit)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
