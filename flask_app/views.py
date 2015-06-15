from flask import render_template, request, send_from_directory
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
    # Make sure to use the same version of features
    #   that the classifier was trained on
    features = make_features1(data)
    # Result of predict is array so get first element
    rating = rating_format[clf.predict(features)[0]]
    probability = np.around(np.max(clf.predict_proba(features)), 2) * 100
    probability = str(int(probability)) + '%'
    return render_template("output_inherited.html", rating=rating,
                           probability=probability, url=url)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)
