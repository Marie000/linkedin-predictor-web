from app import app
from flask import render_template, redirect, url_for, request
from app.forms import DescriptionForm
from app.model import predict


@app.route("/", methods=["GET", "POST"])
def description_page():
    form = DescriptionForm()
    if form.validate_on_submit():
        description = form.description.data
        result = predict(description)
        return redirect(url_for("result_page", result=result))
    return render_template("index.html", form=form)


@app.route("/result")
def result_page():
    result = request.args["result"]
    return render_template("result.html", result=result)
