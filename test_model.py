from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    output_attentions=True
)
model.eval()

img = Image.open("test.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
pred_class = logits.argmax(-1).item()
print("Predicted:", model.config.id2label[pred_class])
print("Number of attention layers:", len(outputs.attentions))
print("Shape of one attention layer:", outputs.attentions[0].shape)

from rollout import attention_rollout

mask = attention_rollout(outputs.attentions)
print("Rollout mask shape:", mask.shape)
print("Min/Max values:", mask.min().item(), mask.max().item())