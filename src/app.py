"""Landmark Explorer API - FastAPI backend for landmark recognition demo."""

import io
import json
import torch
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
EMBEDDINGS_CACHE = CACHE_DIR / "image_embeddings.pt"
METADATA_CACHE = CACHE_DIR / "image_metadata.json"

# Global state (populated at startup)
model = None
processor = None
classifier = None
label_map = None
image_embeds = None
image_paths = []
image_landmarks = []
landmark_to_images = {}
wiki_context = {}


class ClassifierHead(nn.Module):
    """MLP classifier head for CLIP embeddings."""
    def __init__(self, embedding_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


def load_models():
    """Load CLIP + LoRA model and classifier."""
    global model, processor, classifier, label_map
    print("Loading CLIP + LoRA model...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = PeftModel.from_pretrained(base_model, str(MODELS_DIR / "clip_lora" / "best_model"))
    model = model.to(DEVICE).eval()
    print(f"CLIP model loaded on {DEVICE}")

    # Load classifier
    print("Loading classifier...")
    checkpoint = torch.load(MODELS_DIR / "classification" / "best_model.pt", map_location=DEVICE, weights_only=True)
    cfg = checkpoint["config"]
    classifier = ClassifierHead(
        embedding_dim=cfg["embedding_dim"],
        hidden_dims=cfg["hidden_dims"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"]
    ).to(DEVICE)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()

    # Load label map
    label_map = pd.read_csv(MODELS_DIR / "classification" / "label_map.csv")
    print(f"Classifier loaded ({cfg['num_classes']} classes)")


def scan_image_folders():
    """Scan landmark folders and build path/name mappings."""
    global image_paths, image_landmarks, landmark_to_images
    image_paths = []
    image_landmarks = []
    landmark_to_images = {}

    image_dir = DATA_DIR / "landmark_images"
    for folder in sorted(image_dir.iterdir()):
        if not folder.is_dir():
            continue
        # Folder format: 0001_Landmark_Name
        parts = folder.name.split("_", 1)
        landmark_name = parts[1].replace("_", " ") if len(parts) > 1 else folder.name

        folder_images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        landmark_to_images[landmark_name] = [str(p) for p in folder_images]

        for img_path in folder_images:
            image_paths.append(str(img_path))
            image_landmarks.append(landmark_name)

    print(f"Found {len(image_paths)} images across {len(landmark_to_images)} landmarks")


def load_or_compute_embeddings():
    """Load cached embeddings or compute them if cache doesn't exist."""
    global image_embeds, image_paths, image_landmarks, landmark_to_images

    # Always scan folders first (fast)
    scan_image_folders()

    # Check for cached embeddings
    if EMBEDDINGS_CACHE.exists() and METADATA_CACHE.exists():
        print("Loading cached embeddings...")
        cache_data = torch.load(EMBEDDINGS_CACHE, weights_only=True)
        with open(METADATA_CACHE) as f:
            metadata = json.load(f)

        # Load cached paths/landmarks and embeddings
        if "paths" in metadata and "landmarks" in metadata:
            image_embeds = cache_data["embeddings"]
            image_paths = metadata["paths"]
            image_landmarks = metadata["landmarks"]
            print(f"Loaded {image_embeds.shape[0]} cached embeddings")
            return
        else:
            print("Cache format outdated, recomputing...")

    # Compute embeddings
    print("Computing image embeddings (this will be cached for next time)...")
    all_embeds = []
    valid_paths = []
    valid_landmarks = []
    batch_size = 32
    skipped = 0

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_landmarks = image_landmarks[i:i + batch_size]

        # Load images, skipping corrupted ones
        images = []
        paths_ok = []
        landmarks_ok = []
        for p, lm in zip(batch_paths, batch_landmarks):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                paths_ok.append(p)
                landmarks_ok.append(lm)
            except Exception as e:
                print(f"  Skipping corrupted image: {p}")
                skipped += 1

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            embeds = model.base_model.get_image_features(pixel_values=inputs["pixel_values"])
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu())
        valid_paths.extend(paths_ok)
        valid_landmarks.extend(landmarks_ok)

        if (i // batch_size) % 50 == 0:
            print(f"  Processed {len(valid_paths)}/{len(image_paths)} images")

    # Update global lists to only include valid images
    image_paths = valid_paths
    image_landmarks = valid_landmarks
    image_embeds = torch.cat(all_embeds, dim=0)
    print(f"Embeddings shape: {image_embeds.shape} (skipped {skipped} corrupted images)")

    # Save cache (embeddings + paths/landmarks)
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save({"embeddings": image_embeds}, EMBEDDINGS_CACHE)
    with open(METADATA_CACHE, "w") as f:
        json.dump({"paths": image_paths, "landmarks": image_landmarks}, f)
    print(f"Cached embeddings to {CACHE_DIR}")


def load_wiki_context():
    """Load Wikipedia context for landmarks."""
    global wiki_context
    wiki_path = DATA_DIR / "wikipedia_context" / "wiki-context.csv"
    if wiki_path.exists():
        df = pd.read_csv(wiki_path)
        wiki_context = dict(zip(df["landmark_name"], df["page_content"]))
        print(f"Loaded wiki context for {len(wiki_context)} landmarks")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    load_or_compute_embeddings()
    load_wiki_context()
    yield


app = FastAPI(title="Landmark Explorer", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/search")
async def search_landmarks(req: SearchRequest):
    """Search landmarks by text description."""
    inputs = processor(text=[req.query], return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        text_embed = model.base_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    similarities = (text_embed.cpu() @ image_embeds.T).squeeze()

    # Aggregate by landmark (max score)
    landmark_scores = {}
    for idx, score in enumerate(similarities):
        landmark = image_landmarks[idx]
        score_val = score.item()
        if landmark not in landmark_scores or score_val > landmark_scores[landmark]:
            landmark_scores[landmark] = score_val

    top = sorted(landmark_scores.items(), key=lambda x: x[1], reverse=True)[:req.top_k]
    return {"results": [{"landmark": name, "score": round(score, 4)} for name, score in top]}


@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...), top_k: int = 5):
    """Classify uploaded image using trained classifier."""
    # Read and process image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Get CLIP embedding
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.base_model.get_image_features(pixel_values=inputs["pixel_values"])
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        # Classify
        logits = classifier(embedding)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k)

    results = []
    for prob, idx in zip(top_probs[0].cpu(), top_indices[0].cpu()):
        landmark_name = label_map[label_map["landmark_idx"] == idx.item()]["landmark_name"].values[0]
        results.append({"landmark": landmark_name, "confidence": round(prob.item(), 4)})

    # Find similar landmarks using embedding similarity
    similarities = (embedding.cpu() @ image_embeds.T).squeeze()
    landmark_scores = {}
    for idx, score in enumerate(similarities):
        landmark = image_landmarks[idx]
        score_val = score.item()
        if landmark not in landmark_scores or score_val > landmark_scores[landmark]:
            landmark_scores[landmark] = score_val

    # Exclude the top prediction and get similar ones
    top_landmark = results[0]["landmark"] if results else None
    similar = sorted(
        [(name, score) for name, score in landmark_scores.items() if name != top_landmark],
        key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "results": results,
        "similar": [{"landmark": name, "score": round(score, 4)} for name, score in similar]
    }


@app.get("/api/landmark/{name}/images")
async def get_landmark_images(name: str):
    """Get all image paths for a landmark."""
    # Try exact match first, then fuzzy
    images = landmark_to_images.get(name, [])
    if not images:
        # Try case-insensitive match
        for lm, imgs in landmark_to_images.items():
            if lm.lower() == name.lower():
                images = imgs
                break

    # Convert to relative paths for frontend
    rel_paths = [str(Path(p).relative_to(BASE_DIR)) for p in images]
    return {"images": rel_paths}


@app.get("/api/landmark/{name}/info")
async def get_landmark_info(name: str):
    """Get Wikipedia info for a landmark."""
    content = wiki_context.get(name, "")
    if not content:
        # Try case-insensitive/fuzzy match
        for lm, ctx in wiki_context.items():
            if lm.lower() == name.lower() or lm.replace("-", " ").lower() == name.lower():
                content = ctx
                break

    return {"landmark": name, "content": content[:2000] if content else "No information available."}


# Serve static files
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(Path(__file__).parent / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
