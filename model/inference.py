import os

MODEL_PATH = os.environ.get(
    'ALZ_MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'best_model.pth')
)


def load_model():
    import torch
    import torch.nn as nn

    MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\project\model\best_model.pth"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")

    obj = torch.load(MODEL_PATH, map_location='cpu')

    # ✅ Define YOUR ACTUAL MODEL (guessed from weights)
    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, 128),  # depends on input size (224x224)
                nn.ReLU(),
                nn.Linear(128, 4)  # adjust if needed
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    model = CustomCNN()

    # Load weights
    sd = obj.get('state_dict', obj)
    model.load_state_dict(sd, strict=False)

    model.eval()
    return model


# ─────────────────────────────────────────────

def preprocess_image(image_path: str):
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    ext = os.path.splitext(image_path)[1].lower()

    if ext == '.dcm':
        import pydicom
        ds = pydicom.dcmread(image_path)
        arr = ds.pixel_array.astype('float32')

        if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
            arr = arr.max() - arr

        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype('uint8')

        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)

        image = Image.fromarray(arr)
    else:
        image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    return transform(image).unsqueeze(0)


# ─────────────────────────────────────────────

def preprocess_clinical(data: dict):
    import torch

    gender = 1.0 if str(data.get('gender', 'M')).upper().startswith('M') else 0.0

    features = [
        float(data.get('age', 60)),
        gender,
        float(data.get('mmse', 25)),
        float(data.get('cdr', 1)),
        float(data.get('etiv', 1450)),
        float(data.get('nwbv', 0.72)),
        float(data.get('educ', 12)),
        float(data.get('ses', 3)),
    ]

    return torch.tensor([features], dtype=torch.float32)


# ─────────────────────────────────────────────

def postprocess_output(output):
    import numpy as np

    # Convert tensor → numpy
    if hasattr(output, 'detach'):
        output = output.detach().cpu().numpy()

    output = np.array(output).squeeze()

    # ✅ Case 1: Binary output
    if output.ndim == 0:
        conf = float(output)   # 🔥 FIX
        return {
            "prediction": "Alzheimer's Detected" if conf > 0.5 else "No Alzheimer's Detected",
            "confidence": float(round(conf * 100, 2)),  # 🔥 FIX
            "stage": "Moderate" if conf > 0.5 else "Normal",
            "risk": "High" if conf > 0.5 else "Low"
        }

    # ✅ Case 2: Multi-class
    probs = np.exp(output) / np.sum(np.exp(output))
    probs = probs.astype(float)   # 🔥 FIX

    idx = int(np.argmax(probs))   # 🔥 FIX

    labels = ["Normal", "MCI", "Mild AD", "Moderate AD"]

    return {
        "prediction": labels[idx],
        "confidence": float(round(probs[idx] * 100, 2)),  # 🔥 FIX
        "stage": labels[idx],
        "risk": "High" if idx > 1 else "Low"
    }