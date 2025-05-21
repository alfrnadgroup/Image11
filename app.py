from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import traceback

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model("model/animal_model.h5")
    print("? Model loaded successfully")
except Exception as e:
    print("? Failed to load model:", e)
    traceback.print_exc()
    model = None

# Replace with your actual class names
class_labels = ['animal', 'bird', 'tree']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    try:
        if request.method == "POST":
            print("?? POST request received")

            if "file" not in request.files:
                print("? No file part in request")
                return render_template("index.html", prediction="No file part")

            file = request.files["file"]
            if file.filename == "":
                print("? Empty filename")
                return render_template("index.html", prediction="No selected file")

            if file:
                # Save uploaded file temporarily
                os.makedirs("uploads", exist_ok=True)
                img_path = os.path.join("uploads", file.filename)
                file.save(img_path)
                print(f"?? Saved image to {img_path}")

                # Load and preprocess the image
                img = image.load_img(img_path, target_size=(150, 150))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                print("?? Image preprocessed")

                # Predict
                if model is None:
                    raise ValueError("Model is not loaded.")

                preds = model.predict(img_array)
                print("?? Prediction done:", preds)

                predicted_index = np.argmax(preds, axis=1)[0]
                predicted_label = class_labels[predicted_index]
                prediction = f"Prediction: {predicted_label}"

                # Cleanup
                os.remove(img_path)

    except Exception as e:
        print("?? Exception in / route:", e)
        traceback.print_exc()
        prediction = "?? Internal error during prediction. Check logs."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
