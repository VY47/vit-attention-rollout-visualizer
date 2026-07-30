import cv2
import torch
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from rollout import attention_rollout
import ollama

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224', output_attentions=True
)
model.eval()

def explain_attention(label, mask_grid):
    max_idx = mask_grid.argmax()
    row, col = max_idx // mask_grid.shape[1], max_idx % mask_grid.shape[1]
    region = "top" if row < 5 else "bottom" if row > 9 else "middle"
    region += "-left" if col < 5 else "-right" if col > 9 else "-center"
    spread = mask_grid.std()

    prompt = f"""A vision transformer classified a webcam image as "{label}". 
Its attention was most concentrated in the {region} of the frame, 
with a spread score of {spread:.4f} (lower = more focused). 
In one short sentence, explain what this suggests about what the model is looking at."""

    response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
else:
    print("Webcam opened successfully.")

frame_count = 0
last_explanation = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from webcam.")
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    pred_class = outputs.logits.argmax(-1).item()
    label = model.config.id2label[pred_class]

    mask = attention_rollout(outputs.attentions)
    grid_size = int(mask.size(0) ** 0.5)
    mask = mask.reshape(grid_size, grid_size).numpy()
    mask = mask / mask.max()

    frame_count += 1
    if frame_count % 30 == 0:
        last_explanation = explain_attention(label, mask)
        print(last_explanation)

    heatmap = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, last_explanation[:60], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("ViT Attention Rollout", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("ViT Attention Rollout", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()