# ViT Attention Rollout — Live Webcam Visualizer with LLM Explanations

This README provides an overview of the ViT Attention Rollout WebApp, which visualizes what a Vision Transformer "looks at" when classifying live webcam frames, and pairs that visualization with a local LLM that explains the model's attention in plain English.

[[Watch the demo](https://drive.google.com/file/d/1dQDLDotH_W4v5_V1dn4MlNxWtq4aRPdN/view?usp=drive_link)]

## Table of Contents

1. [Introduction](https://github.com/VY47/vit-attention-rollout-visualizer#introduction)
2. [Features](https://github.com/VY47/vit-attention-rollout-visualizer#features)
3. [System Architecture](https://github.com/VY47/vit-attention-rollout-visualizer#system-architecture)
4. [Example Run](https://github.com/VY47/vit-attention-rollout-visualizer#example-run)
5. [Known Limitations](https://github.com/VY47/vit-attention-rollout-visualizer#known-limitations)
6. [Run It Yourself](https://github.com/VY47/vit-attention-rollout-visualizer#run-it-yourself)
7. [References](https://github.com/VY47/vit-attention-rollout-visualizer#references)

## Introduction

The ViT Attention Rollout Visualizer combines a Vision Transformer's internal self-attention mechanics with a local large language model to make model reasoning interpretable in real time. Rather than treating the classifier as a black box, this project implements Attention Rollout (Abnar & Zuidema, 2020) directly from the research paper, then layers a generative model on top to translate raw attention weights into a human-readable explanation. It was built independently, to understand transformer internals hands-on, beyond theory.

## Features

* **Real-Time Attention Visualization**: Overlays a live heatmap on webcam video showing which image regions the ViT is focusing on.
* **From-Paper Implementation**: Attention Rollout implemented directly from Abnar & Zuidema (2020), not a pre-built library.
* **Local LLM Explanation Layer**: A local LLM (Llama 3.2, via Ollama) periodically translates the attention pattern into a plain-English sentence, entirely offline.
* **Discriminative + Generative Pipeline**: Combines a Vision Transformer (discriminative) with an LLM (generative) into a single interpretability tool.
* **Fully Offline**: No cloud API calls or keys required. Runs entirely on local CPU.
* **Transparent About Limitations**: Documents and explains model misclassifications rather than hiding them.

## System Architecture

### Components

1. **Input Capture**: OpenCV captures live frames from the webcam.
2. **Vision Transformer**: `google/vit-base-patch16-224` (via PyTorch and Hugging Face Transformers) splits each frame into 16x16 patches, embeds them with a learnable CLS token, and processes them through 12 transformer encoder layers.
3. **Attention Rollout Engine**: Recursively multiplies per-layer attention matrices, accounting for residual connections, to trace how much each patch ultimately contributed to the CLS token's final classification decision.
4. **Explanation Layer**: A local LLM (Llama 3.2 1B, via Ollama) receives a structured summary of the attention pattern (dominant region, spread) and generates a plain-English explanation every ~30 frames.
5. **Visualization Overlay**: The attention map is rendered as a heatmap and composited onto the live video feed alongside the predicted label and generated explanation.

## Example Run

| Object          | Predicted Label          | Attention Region |
|------------------|---------------------------|--------------------|
| Computer mouse    | mouse                     | center             |
| Phone             | cellphone                 | middle-left        |
| T-shirt (worn)     | jersey / tee shirt        | middle-left        |
| Book              | spatula (misclassified)   | lower-center       |

Note the book misclassification: ImageNet's narrow book category didn't match a book held flat toward the camera, so the model fell back to a shape-similar class. Notably, the LLM's attention explanation remained accurate even when the classification was wrong. It correctly described where the model was looking and why, independent of whether the label itself was right. This distinction (attention fidelity vs. classification accuracy) is part of what makes this project useful as a small interpretability tool, not just a classifier demo.

## Known Limitations

* Uses `google/vit-base-patch16-224`, an ImageNet-pretrained classifier. Its 1000 classes are narrow, specific object categories, not open-world scenes, so predictions on faces, rooms, or uncommon object angles are often inaccurate.
* Runs at a few frames per second on CPU. No GPU optimization was in scope for this build.
* The LLM explanation updates every ~30 frames (not every frame), to keep the video loop responsive since local LLM inference adds latency.

## Run It Yourself

1. Install [Ollama](https://ollama.com) and pull the model:
   ```
   ollama pull llama3.2:1b
   ```
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   python app.py
   ```
4. Press `q` or `Esc` to quit.

## References

* Abnar & Zuidema, ["Quantifying Attention Flow in Transformers"](https://arxiv.org/abs/2005.00928) (2020)
* Dosovitskiy et al., ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) (2020)
