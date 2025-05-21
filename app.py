from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model once when the app starts
model = tf.keras.models.load_model("model/animal_model.h5")

# This should match the classes your model was trained on
class_labels = ['animal', 'bird', 'tree']  # Replace with actual class order if needed

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No selected file"
            else:
                try:
                    # Save uploaded file temporarily
                    img_path = os.path.join("uploads", file.filename)
                    os.makedirs("uploads", exist_ok=True)
                    file.save(img_path)

                    # Load and preprocess the image
                    img = image.load_img(img_path, target_size=(150, 150))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict
                    preds = model.predict(img_array)
                    predicted_index = np.argmax(preds, axis=1)[0]
                    predicted_label = class_labels[predicted_index]

                    prediction = f"Prediction: {predicted_label}"

                    os.remove(img_path)

                except Exception as e:
                    error = f"Prediction failed: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
