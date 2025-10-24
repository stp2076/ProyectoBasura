from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "ClasificacionResiduos.h5"
model = load_model(MODEL_PATH)

# Clases del modelo
CLASSES = ["CARTON", "METAL", "PAPEL", "PLASTICO", "VIDRIO"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Servidor de clasificación activo ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["file"]
    img_path = "temp.jpg"
    file.save(img_path)

    # Preprocesamiento de la imagen
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predicción
    preds = model.predict(x)
    label = CLASSES[np.argmax(preds)]

    return jsonify({"class": label})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))