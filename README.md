# Landmark Explorer

A comprehensive landmark recognition and exploration platform that allows users to find and get information on the perfect landmark destination for their needs.

## What it Does

Landmark Explorer is an intelligent landmark recognition and information retrieval system built on 938 of the world's most popular landmarks. For users who want to discover new landmarks or have heard about some destination but can't exactly recall what it is, we allow putting in custom text descriptions and our contrasively fine-tuned CLIP will find the top 5 landmarks matching the query. Users can then explore these landmarks through a gallery of images as well as a Q&A interface that gets responses from our RAG-based transformer with Wikepedia-informed context. Users can also upload images to get instant landmark identification with confidence scores. Landmark Explorer is a multi-modal system that will enhance users' understanding of the world's most iconic places in a fun, seamless way.

## Quick Start

### Prerequisites


#### Setup

```bash
git clone https://github.com/amoghmanral/landmarks.git
cd landmarks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need (get from Amogh):
- `data/landmark_images/` - the landmark images dataset
- `data/wikipedia_context/wiki-context.csv` - Wikipedia content for RAG

## Run

```bash
python src/app.py
```

Navigate to http://localhost:8000

The application will automatically:
- Load the CLIP + LoRA model and classifier
- Compute or load cached image embeddings
- Load Wikipedia context and RAG data
- Start the FastAPI server with the web interface

**Note:** On first run, image embeddings will be computed and cached for faster subsequent startups.

## Video Links

- **Demo Video:** [Link to demo video]
- **Technical Walkthrough:** [Link to technical walkthrough video]

## Evaluation

### Classification Performance


### Contrasively Fine-Tuned CLIP Performance


### Qualitative Results



## Individual Contributions

- **Aarsh Roongta:**
- **Amogh Manral:**
