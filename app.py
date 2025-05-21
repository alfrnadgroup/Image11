from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/animal_model.h5")
class_labels = ['animal', 'bird', 'tree']  # Make sure this matches your training labels

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            os.makedirs("uploads", exist_ok=True)
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            predicted_index = np.argmax(preds)
            prediction = f"Prediction: {class_labels[predicted_index]}"

            os.remove(filepath)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
