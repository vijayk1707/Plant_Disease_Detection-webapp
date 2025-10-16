import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
from rembg import remove
from flask import Flask, render_template, request, jsonify
import io, os

# --------------------------- Flask setup
app = Flask(__name__, template_folder='templates', static_folder='static')

# --------------------------- Paths
MODEL_PATH = "mobilenetv2_best.pth"
CLASS_NAMES_FILE = "class_names.txt"

# --------------------------- Load class names
with open(CLASS_NAMES_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# --------------------------- Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --------------------------- Image preprocessing
def preprocess_image(image_bytes, brightness_factor=1.2, contrast_factor=1.2, remove_bg=True):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if remove_bg:
        img = remove(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

# --------------------------- Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "Empty file"}), 400

    image_bytes = file.read()
    input_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probs, dim=0)

    predicted_label = class_names[top_class.item()]
    confidence = round(top_prob.item() * 100, 2)

    return jsonify({
        "predicted_class": predicted_label,
        "confidence": confidence
    })

# --------------------------- Run server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)




