from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load trained model
model = load_model("cifar10_model.keras")

# CIFAR-10 class labels
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        # Open image
        img = Image.open(file).convert('RGB')

        # Resize to CIFAR-10 size (32x32)
        img = img.resize((32, 32))

        # Convert to array
        img = np.array(img)

        # Normalize
        img = img / 255.0

        # Reshape (1, 32, 32, 3)
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img)
        class_index = np.argmax(pred)

        result = classes[class_index]

        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction="No file uploaded")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)