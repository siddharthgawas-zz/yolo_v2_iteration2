# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:22:01 2018

@author: NH
"""


import os
from flask import Flask, request, redirect, url_for,jsonify,render_template,flash
from werkzeug.utils import secure_filename
from pred_image import Predict

UPLOAD_FOLDER = r'D:\TV'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/put', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            obj_predict = Predict()
            #prediction = obj_predict.predict_class(UPLOAD_FOLDER,filename)
            prediction = obj_predict.predict_disease(os.path.join(UPLOAD_FOLDER, filename))
            return jsonify(prediction)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
@app.route('/getindices')
def get_fruit_indices():
    return jsonify({'Apple': 0,
 'Apricot': 1,
 'Avocado': 2,
 'Banana': 3,
 'Cocos': 4,
 'Dates': 5,
 'Grape': 6,
 'Guava': 7,
 'Kiwi': 8,
 'Lemon': 9,
 'Limes': 10,
 'Lychee': 11,
 'Mango': 12,
 'Orange': 13,
 'Papaya': 14,
 'Peach': 15,
 'Pineapple': 16,
 'Plum': 17,
 'Pomegranate': 18,
 'Raspberry': 19,
 'Strawberry': 20,
 'Walnut': 21})
    
if __name__ == '__main__':
    app.secret_key = "LOL Nothing so secret here"
    app.run(host = "0.0.0.0",debug=True)