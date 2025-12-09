# Landmark Explorer

A comprehensive landmark recognition and exploration platform that allows users to find and get information on the perfect landmark destination for their needs.

## What it Does

Landmark Explorer is an intelligent landmark recognition and information retrieval system built on 938 of the world's most popular landmarks. For users who want to discover new landmarks or have heard about some destination but can't exactly recall what it is, we allow putting in custom text descriptions and our contrastively fine-tuned CLIP will find the top 5 landmarks matching the query. Users can then explore these landmarks through a gallery of images as well as a Q&A interface that gets responses from our RAG-based transformer with Wikipedia-informed context. Users can also upload images to get instant landmark identification with confidence scores. Landmark Explorer is a multi-modal system that will enhance users' understanding of the world's most iconic places in a fun, seamless way.

## Quick Start

After cloning the repo,

```bash
pip install -r requirements.txt
```

Then get the landmark images zip from ([this Box link](https://duke.box.com/s/w72a997xljs1haztpb3c9fzjr523fmb7)), unzip and place contents into:
```
data/images/landmark_images/
```

### Run

```bash
python src/app.py
```

More detailed instructions can be found in `SETUP.md` if needed.

## Video Links

- **Demo Video:** [[Link to demo video](https://youtu.be/n7J3Orb0Avo)]
- **Technical Walkthrough:** [[Link to technical walkthrough video](https://youtu.be/uo4qSVsYI0A)]

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

**Task:** Given a text description, retrieve the correct landmark from a pool of 938 landmarks.

**Approach:** LoRA fine-tuning on CLIP attention layers with contrastive loss.

| Metric | Base CLIP | LoRA Fine-tuned | Improvement |
|--------|-----------|-----------------|-------------|
| Top-1  | 30.8%     | 33.2%           | +2.4%       |
| Top-5  | 57.0%     | 61.2%           | +4.2%       |
| Top-10 | 69.0%     | 70.6%           | +1.6%       |

*Top-K accuracy: correct landmark appears in the top K results ranked by similarity*

### RAG-based Transformer

**Task:** Given a landmark and query, provide users with a relevant, factually correct answer.

**Approach:** Retrieve the most relevant Wikipedia chunks as context and generate answer using BAAI/bge-base-en-v1.5 transformer. These answers are then graded by OpenAI on a set of objective keywords that should appear in the answer.

| Model | Test Accuracy |
|-------|---------------|
| Baseline (no context) | 74.0% |
| **With RAG context** | **81.0%** |

## Individual Contributions

- **Aarsh Roongta:** Wikipedia descriptions collection, combining images + descriptions into one dataset with train/val/test split, Wikipedia pages and content collection for RAG, chunking + embeddings for retrieval, output generation using transformer
- **Amogh Manral:** Image dataset collection and curation (938 landmarks, 40 images each), MLP classifier training on frozen CLIP embeddings with data augmentation, contrastive fine-tuning of CLIP using LoRA on attention layers, app frontend and backend setup