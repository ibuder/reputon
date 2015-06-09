from flask import render_template, request
from flask_app import app
import pickle
import numpy as np

from get_airbnb_data import make_airbnb_json_dataframe, airbnb_url_to_id
from get_airbnb_data import get_airbnb_by_id
from learning import make_features1

@app.route('/')
@app.route('/input')
def input_():
    return render_template("input.html")

clf = pickle.load(open('pipe1.pkl', 'rb'))
rating_format = {'5-': '5', '4.5-': '4.5', '4-': '4 or lower'}

@app.route('/output')
def output():
    url = request.args.get('URL')
    id_ = airbnb_url_to_id(url)
    if not id_:
        return render_template("output_inherited.html",
                               error='Cannot find Airbnb listing id in URL')
    data = make_airbnb_json_dataframe(get_airbnb_by_id(id_))
    if data is None:
        return render_template("output_inherited.html",
                               error='No data for listing ' + id_)
    features = make_features1(data)
    rating = rating_format[clf.predict(features)[0]]
    probability = np.max(clf.predict_proba(features))
    return render_template("output_inherited.html", rating=rating,
                           probability=probability)
