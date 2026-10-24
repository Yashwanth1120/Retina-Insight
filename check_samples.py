import os, cv2, torch, numpy as np, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, 5))
model.load_state_dict(torch.load("best_retina_model.pth", map_location="cpu", weights_only=True))
model.eval()

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def prep(bgr):
    bgr = cv2.resize(bgr, (256, 256))
    blur = cv2.GaussianBlur(bgr, (0, 0), 10)
    out = cv2.addWeighted(bgr, 4, blur, -4, 128)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

cm = np.zeros((5, 5), dtype=int)
for root, _, files in os.walk("sample_images"):
    for f in sorted(files):
        if f.lower().startswith("stage_") and f.lower().endswith((".png", ".jpg", ".jpeg")):
            true = int(f.lower().split("_")[1][0])
            img = cv2.imread(os.path.join(root, f))
            with torch.no_grad():
                pred = int(model(tf(prep(img)).unsqueeze(0)).argmax(1))
            cm[true, pred] += 1
            print(f"{f:30s} true={true} pred={pred} {'OK' if true == pred else 'WRONG'}")

print("\nConfusion matrix (rows = true stage, columns = predicted):\n", cm)
print("Accuracy:", cm.trace() / cm.sum())
