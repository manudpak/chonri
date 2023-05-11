from flask import Flask, jsonify
import os

# pip3 install transformers
# pip install -q emoji pythainlp==2.2.4 sefr_cut tinydb seqeval sentencepiece pydantic jsonlines
# pip3 install --no-deps thai2transformers==0.1.2
# pip3 install torch

from flask import Flask, jsonify, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

labels_list = ['ถนน           ', 'ความสะอาด', 'แสงสว่าง   ', 'ความปลอดภัย', 'จราจร             ', 'อื่น ๆ      ']

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>สวัสดี</h1>"

@app.route('/get_label', methods=['POST'])
def get_label():
    #get data from app
    txt = request.get_json()
    msg = txt['input'] 
    print(msg)

    # Load the saved model
    model_path = "/model" # forgot change path!!!
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Preprocess the input text
    encoded_input = tokenizer(msg, truncation=True, padding=True, max_length=416, return_tensors='pt')

    # Make predictions
    with torch.no_grad():
        logits = model(**encoded_input).logits
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.cpu().numpy()[0]

    # Create a dictionary of labels and probabilities
    label_prob_dict = {}
    for i, label in enumerate(labels_list):
        label_prob_dict[label] = probabilities[i]

    # Sort the dictionary by probability in descending order
    label_prob_dict = {k: v/6 for k, v in sorted(label_prob_dict.items(), key=lambda item: item[1], reverse=True)}

    # Normalize the probabilities and ensure that the sum is 100
    prob_sum = sum(label_prob_dict.values())
    label_prob_dict = {k: v/prob_sum for k, v in label_prob_dict.items()}

    results = {k: round(v, 4) for k, v in label_prob_dict.items()}
    print(results)

    data = results
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
