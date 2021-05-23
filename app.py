from flask import Flask, redirect, url_for, request, render_template
import pickle
# import model


detection_model = pickle.load(open('classifier.pkl', 'rb'))

app = Flask(__name__)


posts = [
    {
        'author':'Anirudh Dutt',
        'title':'Stock Prediction',
        'content': 'First post',
        'date_posted':'April 2021'
    },
    {
        'author':'Anirudh Dutt',
        'title':'Stock Prediction',
        'content': 'second post',
        'date_posted':'may 2021'
    }
]


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/trend', methods=['GET'])
def trend():
    return render_template("trend.html")


@app.route('/trend', methods=['POST'])
def detect_trend():
    rsi = request.form['rsi']
    crc = request.form['crc']
    obv = request.form['obv']
    vwap = request.form['vwap']
    res = detection_model.predict([[rsi, crc, obv, vwap]])
    return render_template('detect_trend.html', data=res)

# @app.route('/send', methods=['GET', 'POST'])
# def predict():
#     if request.method== 'POST':
#         prediction = predict_price()
    
#     if prediction!=0:
#         return render_template("index.html", prediction = prediction)

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    return render_template("visualize.html", posts=posts)

@app.route('/projectlink', methods=['GET', 'POST'])
def projectlink():
    return render_template("projectlink.html", posts=posts)


if __name__ == "__main__":
    app.run(debug=True)