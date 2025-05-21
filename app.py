import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/animal_model.h5")

# Class labels
class_labels = ['animal', 'bird', 'tree']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No selected file")

        if file:
            file_path = "temp.jpg"
            file.save(file_path)

            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            label = class_labels[np.argmax(preds)]

            prediction = f"Prediction: {label}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ? Use Render's assigned port
    app.run(host="0.0.0.0", port=port)        # ? Bind to 0.0.0.0
