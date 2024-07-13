import os
from flask import Flask, request, render_template
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
import json

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = models.mobilenet_v2(pretrained=True)
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

# Load the ImageNet class index file
with open('imagenet_classes.json') as f:
    class_idx = json.load(f)
    idx2label = {int(key): value for key, value in class_idx.items()}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    img_file = request.files['file']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(img_path)

    img = Image.open(img_path)
    img_t = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        preds = model(img_t)
    probabilities = torch.nn.functional.softmax(preds[0], dim=0)

    # Get the top 3 predicted classes
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    results = [(idx2label[int(c)], float(p)) for c, p in zip(top3_catid, top3_prob)]

    return {"predictions": results}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
