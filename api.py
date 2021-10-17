from flask import Flask, request, jsonify
from model import predictAlphabet

app = Flask(__name__)


@app.route("/alphabetPrediction", methods=['POST'])
def predict():
    # first we have to get the image file by post in our api by a key called "letter"
    # we are using the key :- letter
    imagefile = request.files.get("letter")

    # done the prediction of the imagefile
    prediction = predictAlphabet(imagefile)

    # returning the prediction in the form of json
    return jsonify({"prediction": prediction, "status": "😊"})


# running the app
if __name__ == '__main__':
    app.run(debug=True)
