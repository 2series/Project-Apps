"""
Routes and views for the flask application.
"""
import os
from datetime import datetime
from flask import render_template, request
from werkzeug import secure_filename
from terror_ai import app
import numpy as np
import pandas as pd
from terror_ai.decision_tree_ml import decision_main
from terror_ai.logistic_regression_ml import logi_main


# Get the real output from the predicted label
def get_real_output(pred_label, filepath, flag):
    temp_df = pd.DataFrame.from_csv(filepath, header=0)
    temp_df = temp_df.reset_index()
    val1 = "NA"
    val2 = "NA"
    if flag == 1:
        for i, row in temp_df.iterrows():
            if row["targtype1_txt_code"] == pred_label[0]:
                val1 = row["targtype1_txt"]
                return val1
    if flag == 2:
        for i, row in temp_df.iterrows():
            if row["attacktype1_txt_code"] == pred_label[0]:
                val2 = row["attacktype1_txt"]
                return val2


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html'
        )


@app.route("/form")
def form():
    return render_template(
        "form.html"
        )


@app.route("/uploader", methods=["GET", "POST"])
def uploader():
    month_val=int(request.form["monthnumber"])
    casualty_val=int(request.form["ncasualty"])
    country_val=int(request.form["countryname"])
    region_val=int(request.form["regionname"])
    target_val=int(request.form["targetname"])
    
    file1 = "/mnt/d/COLLEGE/DONE/terror_ai/terror_ai/static/data/target_type.csv"
    file2 = "/mnt/d/COLLEGE/DONE/terror_ai/terror_ai/static/data/attack_type.csv"

    if int(request.form["algo"]) == 1:
        algorithm_to_use = "Decision Tree"

        df1 = pd.DataFrame.from_csv("/mnt/d/COLLEGE/DONE/terror_ai/terror_ai/static/data/final_train1.csv", header=0)
        # form a test vector
        test_df1 = pd.DataFrame([month_val, casualty_val, country_val, region_val, target_val])
        test_df1 = test_df1.transpose()
        # train and predict using model1
        pred_label1, accuracy_value1 = decision_main(df1, test_df1)
        #pred_label1, accuracy_value1 = decision_main(train1X, train1y, test_df1)
        out1 = get_real_output(pred_label1, file1, 1)

        df2 = pd.DataFrame.from_csv("/mnt/d/COLLEGE/DONE/terror_ai/terror_ai/static/data/final_train2.csv", header=0)
        # form a test vector
        test_df2 = pd.DataFrame([month_val, casualty_val, country_val, region_val, target_val, pred_label1])
        test_df2 = test_df2.transpose()
        # train and predict using model1
        pred_label2, accuracy_value2 = decision_main(df2, test_df2)
        out2 = get_real_output(pred_label2, file2, 2)
    else:
        algorithm_to_use = "Logistic Regression"

        df1 = pd.DataFrame.from_csv("/mnt/d/COLLEGE/DONE/terror_ai/terror_ai/static/data/final_train1.csv", header=0)
        # form a test vector
        test_df1 = pd.DataFrame([month_val, casualty_val, country_val, region_val, target_val])
        test_df1 = test_df1.transpose()
        # train and predict using model1
        pred_label1, accuracy_value1 = logi_main(df1, test_df1)
        out1 = get_real_output(pred_label1, file1, 1)

        df2 = pd.DataFrame.from_csv("/mnt/d/COLLEGE/DONE/terror_ai/terror_ai/static/data/final_train2.csv", header=0)
        # form a test vector
        test_df2 = pd.DataFrame([month_val, casualty_val, country_val, region_val, target_val, pred_label1])
        test_df2 = test_df2.transpose()
        # train and predict using model1
        pred_label2, accuracy_value2 = logi_main(df2, test_df2)
        out2 = get_real_output(pred_label2, file2, 2)
    return render_template(
        "output.html",
        alg=algorithm_to_use,
        res1=out1,
        res2=out2,
        ac1=accuracy_value1,
        ac2=accuracy_value2,
        country=country_val,
        casualty=casualty_val
    )