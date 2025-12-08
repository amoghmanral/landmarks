# Landmark Explorer

A comprehensive landmark recognition and exploration platform that allows users to find and get information on the perfect landmark destination for their needs.

## What it Does

Landmark Explorer is an intelligent landmark recognition and information retrieval system built on 938 of the world's most popular landmarks. For users who want to discover new landmarks or have heard about some destination but can't exactly recall what it is, we allow putting in custom text descriptions and our contrastively fine-tuned CLIP will find the top 5 landmarks matching the query. Users can then explore these landmarks through a gallery of images as well as a Q&A interface that gets responses from our RAG-based transformer with Wikipedia-informed context. Users can also upload images to get instant landmark identification with confidence scores. Landmark Explorer is a multi-modal system that will enhance users' understanding of the world's most iconic places in a fun, seamless way.

## Quick Start

### Prerequisites

- Python 3.10+
- ~8GB RAM (for transformer model)

### Setup

```bash
git clone https://github.com/amoghmanral/landmarks.git
cd landmarks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then get the landmark images zip from (insert box link here later), unzip and place contents into:
```
data/images/landmark_images/
```

### Run

```bash
python src/app.py
```

Navigate to http://localhost:8000

The application will automatically:
- Load the CLIP + LoRA model and classifier
- Load cached image embeddings (included in repo)
- Load Wikipedia context and RAG data
- Download the Qwen transformer model (~3GB, first run only)
- Start the FastAPI server with the web interface

## Video Links

- **Demo Video:** [Link to demo video]
- **Technical Walkthrough:** [Link to technical walkthrough video]

## Evaluation

### Classification Performance

**Task:** Given an image, classify it as one of 938 landmark classes.

**Approach:** MLP classifier head trained on frozen CLIP ViT-B/32 embeddings with data augmentation.

| Model | Test Accuracy |
|-------|---------------|
| Zero-shot CLIP | 64.69% |
| **Our Classifier** | **82.69%** |

*+18% improvement over zero-shot baseline*

### Contrastively Fine-Tuned CLIP Performance


### Qualitative Results



## Individual Contributions

- **Aarsh Roongta:**
- **Amogh Manral:** Image dataset collection and curation, CLIP classifier training, contrastive CLIP fine-tuning (LoRA), web application frontend and backend
