import os
from flask import Flask, flash, request, redirect, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from joblib import dump, load
from utils import import_data, preprocess



app = Flask(__name__)
enc = load("encoder.pkl")
model = load("model.pkl")
UPLOAD_FOLDER = ""
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif", "json"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in "json"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        print(file)
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            print("---", file)
            flash("No selected file")
            return redirect(request.url)
        if file:  # and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            df = pd.read_json(file)
            df = preprocess(df)
            cat_cols = df.select_dtypes(include=["object"]).columns
            df[cat_cols] = enc.transform(df[cat_cols].values)
            predict = model.predict(df)
            pred = np.where((predict==0)|(predict==1), predict^1, predict)

            pd.Series(pred).to_json("results.json", orient="records")
            return send_file(
                "results.json",
                mimetype="application/json",
                attachment_filename="results.json",
                as_attachment=True,
            )

    return """
    <!doctype html>
    <center>
    <title>Catching Joe</title>
    <h1>Upload new test set</h1>
    <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </center>
    """


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv('PORT', 4444)))
