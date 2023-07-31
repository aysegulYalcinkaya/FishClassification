import pickle

import pandas as pd
import numpy as np
import sklearn

from flask import Flask, request, render_template

app = Flask(__name__)


def classify_input(input_data):
    with open('fish_classifier.pkl', 'rb') as f:
        classifier_model = pickle.load(f)

    result = classifier_model.predict(input_data)

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs from the form
        weight = request.form['weight']
        length1 = request.form['length1']
        length2 = request.form['length2']
        length3 = request.form['length3']
        height = request.form['height']
        width = request.form['width']

        # Perform classification using the inputs
        input_data = pd.DataFrame(np.array([weight, length1, length2, length3, height, width]).reshape(1, -1),
                                  columns=["Weight", "Length1", "Length2", "Length3", "Height", "Width"])
        result = classify_input(input_data)

        return render_template('index.html', result=result)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
