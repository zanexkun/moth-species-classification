import torch
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io


# ─────────────────────────────────────────
# Model setup — runs once when server starts
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 50)
model.load_state_dict(torch.load("moth1.pth", map_location=device))
model.to(device)
model.eval()

# Alphabetical order — matches ImageFolder assignment during training
class_names = [
    "ARCIGERA FLOWER MOTH", "ATLAS MOTH", "BANDED TIGER MOTH",
    "BIRD CHERRY ERMINE MOTH", "BLACK RUSTIC MOTH", "BLAIRS MOCHA",
    "BLOTCHED EMERALD MOTH", "BLUE BORDERED CARPET MOTH", "CINNABAR MOTH",
    "CLEARWING MOTH", "COMET MOTH", "DEATHS HEAD HAWK MOTH",
    "ELEPHANT HAWK MOTH", "EMPEROR GUM MOTH", "EMPEROR MOTH",
    "EYED HAWK MOTH", "FIERY CLEARWING MOTH", "GARDEN TIGER MOTH",
    "HERCULES MOTH", "HORNET MOTH", "HUMMING BIRD HAWK MOTH",
    "IO MOTH", "JULY BELLE MOTH", "KENTISH GLORY MOTH",
    "LACE BORDER MOTH", "LEOPARD MOTH", "LUNA MOTH",
    "MADAGASCAN SUNSET MOTH", "MAGPIE MOTH", "MUSLIN MOTH",
    "OLEANDER HAWK MOTH", "OWL MOTH", "PALPITA VITREALIS MOTH",
    "PEACH BLOSSOM MOTH", "PLUME MOTH", "POLYPHEMUS MOTH",
    "PURPLE BORDERED GOLD MOTH", "RED NECKED FOOTMAN MOTH", "REGAL MOTH",
    "ROSY MAPLE MOTH", "ROSY UNDERWING MOTH", "RUSTY DOT PEARL MOTH",
    "SCHORCHED WING MOTH", "SIXSPOT BURNET MOTH", "SQUARE SPOT RUSTIC MOTH",
    "SUSSEX EMERALD MOTH", "SWALLOW TAILED MOTH", "VESTAL MOTH",
    "WHITE LINED SPHINX MOTH", "WHITE SPOTTED SABLE MOTH",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────
# API
# ─────────────────────────────────────────
app = FastAPI()


@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    contents = await img.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    image    = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output        = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = probabilities.max(1)

    return {
        "prediction": class_names[prediction.item()],
        "confidence": round(confidence.item() * 100, 2),
    }
