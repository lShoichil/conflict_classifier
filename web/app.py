from flask import Flask, render_template, request
import torch
import pickle
from model import Model, predict

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''
    if request.method == 'POST':
        model = Model()
        model.state_dict(torch.load("toxificator-512.pth", map_location=torch.device('cpu')))
        model.eval()
        user_input = request.form.get('user_input_text')
        prediction = predict(model, user_input)
        print(prediction)
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
