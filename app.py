from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 🔹 Desactiva CUDA porque Render no tiene GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
CORS(app)  # 🔹 Permite peticiones desde tu HTML (para evitar errores CORS)

# 🔹 Carga del modelo
MODEL_PATH = "ClasificacionResiduos.h5"
model = load_model(MODEL_PATH)

# 🔹 Clases del modelo
CLASSES = ["CARTON", "METAL", "PAPEL", "PLASTICO", "VIDRIO"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Servidor de clasificación activo ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400

        file = request.files["file"]

        # 🔹 Guarda temporalmente la imagen
        img_path = "temp.jpg"
        file.save(img_path)

        # 🔹 Preprocesamiento
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # 🔹 Predicción
        preds = model.predict(x)
        label = CLASSES[np.argmax(preds)]

        return jsonify({"class": label})

    except Exception as e:
        # 🔹 Devuelve el error al frontend para depuración
        print("Error en /predict:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 🔹 Render asigna su propio puerto con la variable de entorno PORT
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))