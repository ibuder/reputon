from flask import render_template, request
from flask_app import app

from get_airbnb_data import make_airbnb_json_dataframe, airbnb_url_to_id
from get_airbnb_data import get_airbnb_by_id


@app.route('/')
@app.route('/input')
def input_():
    return render_template("input.html")


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
    return render_template("output_inherited.html")
