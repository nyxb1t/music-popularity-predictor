from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        features = [
            float(request.form["danceability"]),
            float(request.form["energy"]),
            float(request.form["loudness"]),
            float(request.form["speechiness"]),
            float(request.form["acousticness"]),
            float(request.form["instrumentalness"]),
            float(request.form["liveness"]),
            float(request.form["valence"]),
            float(request.form["tempo"]),
            float(request.form["explicit"])
        ]

        final_input = scaler.transform([features])
        prediction = model.predict(final_input)[0]
        prediction_value = round(float(prediction), 2)

        return render_template(
            "predict.html",
            prediction=prediction_value,
            values=request.form
        )

    return render_template("predict.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)