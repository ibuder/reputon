from flask import render_template, request, send_from_directory
from flask_app import app
import pickle
import numpy as np
import pandas as pd

from get_airbnb_data import make_airbnb_json_dataframe, airbnb_url_to_id
from get_airbnb_data import get_airbnb_by_id
from learning import make_features5
import settings
import db


@app.route('/')
@app.route('/input')
def input_():
    return render_template("input.html")

clf = pickle.load(open('pipe2.pkl', 'rb'))
rating_format = {'5-': '5', '4.5-': '4.5', '4-': '4 or lower',
                 '4.75+': '5 (this might be a great deal!)',
                 '4.75-': '4.5 or lower'}


make_features = make_features5  # Make sure this matches classifier in use


@app.route('/output')
def output():
    url = request.args.get('URL')
    id_ = airbnb_url_to_id(url)
    if not id_:
        return render_template("output_inherited.html",
                            error='Cannot find Airbnb listing id in given URL')
    data = make_airbnb_json_dataframe(get_airbnb_by_id(id_))
    if data is None:
        return render_template("output_inherited.html",
                               error='No data available for listing ' + id_)
    assert len(data) == 1
    data.set_index('id', inplace=True)  # needed to make features
    # Make sure to use the same version of features
    #   that the classifier was trained on
    features = make_features(data)
    assert len(features) == 1
    # Result of predict is array so get first element
    rating = rating_format[clf.predict(features)[0]]
    probability = np.around(np.max(clf.predict_proba(features)), 2) * 100
    probability = str(int(probability)) + '%'

    important_columns = feature_importances(features)

    return render_template("output_inherited.html", rating=rating,
                           probability=probability, url=url,
                           embedly_key=settings.embedly,
                           feature=features.columns[important_columns[0]])


n_comparison_examples = 100  # Examples for calculating feature importance
engine = db.create_root_engine()
# Need to reset index because we are combining data with different id
#   so don't want to align by index
comparison_examples = make_features(pd.io.sql.read_sql(
    'SELECT * FROM listings ORDER BY RAND() LIMIT {0}'.format(
    n_comparison_examples), engine, index_col='id')).reset_index(drop=True)


# TODO take the predict_proba out of loop
def feature_importances(example):
    """
    Returns sorted indices of most important features for this example

    example: single row of features
    """
    single_proba = clf.predict_proba(example)
    class_ = np.argmax(single_proba)  # index of predicted class
    importances = []
    n_columns = len(example.columns)
    scrambled_examples = example.iloc[
        np.repeat(0, n_comparison_examples * n_columns),
        :].reset_index(drop=True)
    for icolumn, column in enumerate(example.columns):
        # Replace this feature with random values
        #   first n_comparison_examples for first column, then
        #   n_comparison_examples for second column
        # Use .values to evade index alignment
        scrambled_examples.iloc[icolumn*n_comparison_examples:
                            (icolumn+1)*n_comparison_examples, icolumn] = \
                            comparison_examples.iloc[:, icolumn].values
    scrambled_probas = clf.predict_proba(scrambled_examples)
    # Either big spread or mean offset of
    #   scrambled predictions - normal prediction
    #   means feature was important
    #   so take RMS
    proba_differences = (scrambled_probas[:, class_] -
                         single_proba[0, class_]) ** 2
    # column changes most slowly so put first (C order)
    proba_differences.shape = (n_columns, n_comparison_examples,)
    importances = np.mean(proba_differences, axis=1)

    # Reverse order to make most important first
    return np.argsort(importances)[::-1]


@app.route('/slides')
def slides():
    return render_template("slides.html")


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)
