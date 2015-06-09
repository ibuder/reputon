from flask import render_template, request
from flask_app import app


@app.route('/')
@app.route('/input')
def input_():
    return render_template("input.html")


@app.route('/output')
def output():
    return render_template("output_inherited.html")
