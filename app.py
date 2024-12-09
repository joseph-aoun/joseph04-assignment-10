import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer

app = Flask(__name__)

MODEL_NAME = "ViT-B/32"
PRETRAINED = "openai"
IMAGE_FOLDER = "static/coco_images_resized"
EMBEDDINGS_PATH = "embeddings/image_embeddings.pickle"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(device)
model.eval()

df = pd.read_pickle(EMBEDDINGS_PATH)
image_embeddings = np.stack(df['embedding'].values)
file_names = df['file_name'].values

import open_clip
text_tokenizer = open_clip.get_tokenizer('ViT-B-32')

def get_text_embedding(text_query):
    text_tokens = text_tokenizer([text_query])
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb = F.normalize(text_emb, p=2, dim=-1)
    return text_emb

def get_image_embedding(image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = model.encode_image(img)
        image_emb = F.normalize(image_emb, p=2, dim=-1)
    return image_emb

def get_top_k_images(query_embedding, image_embeddings, file_names, k=5):
    query_vec = query_embedding.cpu().numpy()  # shape: (1, D)
    similarities = np.dot(query_vec, image_embeddings.T).flatten()
    top_k_indices = np.argpartition(similarities, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    top_files = file_names[top_k_indices]
    top_scores = similarities[top_k_indices]
    return list(zip(top_files, top_scores))

def get_pca_transformed_embeddings(image_embeddings, k):
    pca = PCA(n_components=k)
    reduced_embeddings = pca.fit_transform(image_embeddings)
    return pca, reduced_embeddings

def project_query_to_pca(query_embedding, pca):
    query_vec = query_embedding.cpu().numpy()
    return pca.transform(query_vec)

def get_top_k_images_pca(query_embedding, reduced_embeddings, file_names, pca, k=5):
    query_pca = project_query_to_pca(query_embedding, pca)
    query_norm = query_pca / np.linalg.norm(query_pca, axis=1, keepdims=True)
    emb_norm = reduced_embeddings / np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
    similarities = np.dot(query_norm, emb_norm.T).flatten()
    top_k_indices = np.argpartition(similarities, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    top_files = file_names[top_k_indices]
    top_scores = similarities[top_k_indices]
    return list(zip(top_files, top_scores))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    text_query = request.form.get("text_query", "").strip()
    lam = float(request.form.get("lambda", "1.0"))
    use_pca = request.form.get("use_pca", "off") == "on"
    k_components = int(request.form.get("k_components", "100"))

    image_file = request.files.get("image_query", None)
    has_image = (image_file and image_file.filename != "")

    text_emb = None
    image_emb = None

    if text_query:
        text_emb = get_text_embedding(text_query)

    if has_image:
        img = Image.open(image_file.stream).convert("RGB")
        image_emb = get_image_embedding(img)

    if text_emb is not None and image_emb is not None:
        query = F.normalize(lam * text_emb + (1.0 - lam) * image_emb, p=2, dim=-1)
    elif text_emb is not None:
        query = text_emb
    elif image_emb is not None:
        query = image_emb
    else:
        return render_template("index.html", error="Please provide a text or image query.")

    if use_pca:
        pca, reduced_embeddings = get_pca_transformed_embeddings(image_embeddings, k_components)
        results = get_top_k_images_pca(query, reduced_embeddings, file_names, pca, k=5)
    else:
        results = get_top_k_images(query, image_embeddings, file_names, k=5)

    return render_template("index.html", results=results, text_query=text_query, lam=lam, use_pca=use_pca, k_components=k_components)

if __name__ == "__main__":
    app.run(debug=True)