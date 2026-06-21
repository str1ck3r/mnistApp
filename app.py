"""
Flask-сервер для распознавания рукописных цифр (MNIST).

Принимает изображение, нарисованное пользователем на canvas (в виде base64 PNG),
сжимает его до 28x28 пикселей, нормализует и подаёт в обученную нейросетевую модель.

Запуск локально:
    python app.py
Сайт будет доступен на http://127.0.0.1:5000
"""

import base64
import io

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# ========== ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ ==========
# Файл mnist_model.keras должен лежать в той же папке, что и app.py
MODEL_PATH = "mnist_model.keras"
model = keras.models.load_model(MODEL_PATH)
print(f"Модель успешно загружена из {MODEL_PATH}")


def preprocess_image(image_data_url: str) -> np.ndarray:
    """
    Преобразует изображение с canvas (base64 PNG, рисунок на белом/прозрачном фоне,
    чёрная линия) в формат, который ожидает модель MNIST:
    массив (1, 28, 28, 1), значения float32 в диапазоне [0, 1],
    белая цифра на чёрном фоне (как в оригинальном датасете MNIST).
    """
    # Отрезаем префикс "data:image/png;base64,"
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    # Открываем как изображение и переводим в градации серого
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # На canvas рисуем чёрным по белому, а MNIST — белым по чёрному.
    # Поэтому инвертируем цвета.
    image = ImageOps.invert(image)

    # Сжимаем до 28x28 с хорошим сглаживанием (антиалиасинг)
    image = image.resize((28, 28), Image.LANCZOS)

    # В numpy-массив и нормализация в [0, 1]
    img_array = np.array(image).astype("float32") / 255.0

    # Форма (1, 28, 28, 1), как ожидает модель
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "Изображение не передано"}), 400

    try:
        img_array = preprocess_image(data["image"])
    except Exception as e:
        app.logger.exception("Ошибка обработки изображения")
        return jsonify({"error": f"Ошибка обработки изображения: {e}"}), 400

    # Предсказание
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    # Вероятности по всем классам — для наглядности на сайте
    all_probabilities = [round(float(p) * 100, 2) for p in predictions]

    return jsonify(
        {
            "digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "probabilities": all_probabilities,
        }
    )


if __name__ == "__main__":
    # debug=True только для локальной разработки, на проде поставить False
    app.run(host="0.0.0.0", port=5000, debug=True)