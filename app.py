from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/animal_model.h5")

# Update based on your training labels
class_labels = ['animal', 'bird', 'tree']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            prediction = "No file part"
        else:
            file = request.files["file"]
            if file.filename == "":
                prediction = "No selected file"
            else:
                # Save uploaded file
                os.makedirs("uploads", exist_ok=True)
                filepath = os.path.join("uploads", file.filename)
                file.save(filepath)

                try:
                    # Preprocess image — adjust target size & channels to match your model
                    img = image.load_img(filepath, target_size=(28, 28), color_mode="grayscale")
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 28, 28, 1)

                    # Predict
                    preds = model.predict(img_array)
                    predicted_index = np.argmax(preds, axis=1)[0]
                    predicted_label = class_labels[predicted_index]
                    prediction = f"Prediction: {predicted_label}"
                except Exception as e:
                    prediction = f"Error during prediction: {str(e)}"
                finally:
                    os.remove(filepath)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
