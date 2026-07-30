\# ViT Attention Rollout — Live Webcam Visualizer with LLM Explanations



A real-time visualisation of what a Vision Transformer "looks at" when

classifying webcam frames, using Attention Rollout (Abnar \& Zuidema, 2020) —

paired with a local LLM that translates the raw attention pattern into a

plain-English explanation.



\## Why I built this



I wanted to understand how self-attention actually produces a classification

decision in a ViT, rather than treating it as a black box — so I implemented

attention rollout myself and visualised it live on my own webcam. I then

added a local LLM (Llama 3.2, via Ollama) on top, to see whether a model's

internal attention pattern could be translated into a human-readable

explanation in real time — effectively building a small interpretability

pipeline that combines a discriminative model (ViT) with a generative one

(LLM).



\## How it works



The image is split into 16x16 patches and embedded, with a learnable CLS

token prepended to the sequence. As this sequence passes through 12

transformer layers, self-attention lets every patch attend to every other

patch. Attention rollout recursively multiplies the attention matrices

across all layers — accounting for residual connections by adding the

identity matrix at each step — to produce a single map showing how much

each patch ultimately contributed to the CLS token's final classification

decision.



A local LLM (Llama 3.2 1B, run entirely offline via Ollama) periodically

receives a structured summary of the attention pattern — which region of

the frame it's concentrated in, and how spread out vs. focused it is — and

generates a one-sentence plain-English explanation of what the model

appears to be focusing on.



\## Demo



https://drive.google.com/file/d/1dQDLDotH\_W4v5\_V1dn4MlNxWtq4aRPdN/view?usp=drive\_link



\## Example run



| Object              | Predicted Label       | Attention Region |

| Computer mouse       | mouse                 | center            |

| Phone                | cellphone              | middle-left       |

| T-shirt (worn)        | jersey / tee shirt     | middle-left       |

| Book                 | spatula (misclassified)| lower-center      |



Note the book misclassification: ImageNet's narrow book category didn't

match a book held flat toward the camera, so the model fell back to a

shape-similar class. Notably, the LLM's \*attention explanation\* remained

accurate even when the \*classification\* was wrong — it correctly described

where the model was looking and why, independent of whether the label

itself was right. This distinction (attention fidelity vs. classification

accuracy) is part of what makes this project useful as a small

interpretability tool, not just a classifier demo.



\## Known limitations



\- Uses `google/vit-base-patch16-224`, an ImageNet-pretrained classifier —

&#x20; its 1000 classes are narrow, specific object categories, not open-world

&#x20; scenes, so predictions on faces, rooms, or uncommon object angles are

&#x20; often inaccurate.

\- Runs at a few frames per second on CPU — no GPU optimization was in

&#x20; scope for this build.

\- The LLM explanation updates every \~30 frames (not every frame), to keep

&#x20; the video loop responsive since local LLM inference adds latency.



\## Run it yourself



1\. Install \[Ollama](https://ollama.com) and pull the model:
    
    ollama pull llama3.2:1b

2. Install Python dependencies:

   pip install -r requirements.txt

3. Run the app:

   python app.py

4. Press `q` or `Esc` to quit.



\## References



\- Abnar \& Zuidema, \["Quantifying Attention Flow in Transformers"](https://arxiv.org/abs/2005.00928) (2020)

\- Dosovitskiy et al., \["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) (2020)



