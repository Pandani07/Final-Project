from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)

MODEL_PATH = 'model_100.h5'

from tensorflow.keras.models import load_model
model = load_model(MODEL_PATH)
print("Tensorflow/ LSTM model loaded {}".format(str(model)))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/trend', methods=['GET', 'POST'])
def trend():
    return render_template("trend.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    return render_template("visualize.html")

@app.route('/projectlink', methods=['GET', 'POST'])
def projectlink():
    return render_template("projectlink.html")


if __name__ == "__main__":
    app.run(debug=True)