import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ArtGAN.task1_crnn.model import ConvRecurrentWikiArtClassifier
from ArtGAN.task1_crnn.train import choose_device

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "ArtGAN_outputs" / "task1_crnn" / "task1_crnn" / "best.pt"
IMAGE_PATH = BASE_DIR / "Lady_with_an_Ermine_-_Leonardo_da_Vinci_(adjusted_levels).jpg"
TOPK = 5

device = choose_device("auto")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

model = ConvRecurrentWikiArtClassifier(
    num_classes=checkpoint["num_classes"],
    pretrained_backbone=checkpoint.get("args", {}).get("pretrained_backbone", False),
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image_size = checkpoint["args"]["image_size"]
crop_size = checkpoint["args"]["crop_size"]

transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

image = Image.open(IMAGE_PATH).convert("RGB")
x = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(x)

def top_predictions(logits, class_names, topk=5):
    probs = torch.softmax(logits, dim=-1)[0]
    values, indices = torch.topk(probs, k=min(topk, probs.shape[0]))
    return [
        {"label": class_names[i], "prob": float(v)}
        for v, i in zip(values.cpu().tolist(), indices.cpu().tolist())
    ]

results = {
    "style": top_predictions(outputs["style"], checkpoint["class_names"]["style"], TOPK),
    "genre": top_predictions(outputs["genre"], checkpoint["class_names"]["genre"], TOPK),
    "artist": top_predictions(outputs["artist"], checkpoint["class_names"]["artist"], TOPK),
}

print(json.dumps(results, indent=2))
