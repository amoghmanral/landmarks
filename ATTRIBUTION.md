# Attribution

## AI-Generated Code

### Data Collection (`data/`)

**`data/images/image_scraper.py`** - Image scraping script
- AI-assisted, particularly `download_images_for_landmark` function
- DuckDuckGo Images scraping syntax and stock image filtering logic generated with AI

**`data/clip/fetch_descriptions.py`** - Description generation script
- Assisted in prompt engineering
- Code for pairing images to descriptions

**`data/clip/split.py`** - Train/Val/Test splitting script
- Code for generating file path to image in the landmark's folder

**`data/rag/fetch_wiki_url.py`** - Fetching Wikipedia url script
- API call syntax

**`data/rag/fetch_content.py`** - Fetching Wikipedia content script
- API call syntax

### Training Notebooks (`notebooks/`)

**`notebooks/clip_classifier.ipynb`** - Classifier training notebook
- Data augmentation pipeline (`train_augment` transforms composition) AI-assisted
- Training loop boilerplate functions AI-assisted
- Plotting code and results formatting AI-assisted


**`notebooks/clip_lora.ipynb`** - LoRA fine-tuning notebook
- LoRA configuration and setup (`LoraConfig`, `get_peft_model`, attention layer targeting) AI-generated
- Contrastive loss function (`contrastive_loss`) AI-assisted
- Training loop (`train_epoch_lora`) AI-assisted
- Evaluation function (`evaluate_lora`) and final base vs LoRA comparison AI-generated
- DataLoader with custom `collate_fn` AI-assisted

**`notebooks/rag.ipynb`** - RAG + transformer notebook
- chunking text AI-generated
- evaluation AI-assisted

### Web Application (`src/`)

**`src/index.html`** - Frontend interface
- Almost entirely AI-generated through iterative prompting

**`src/app.py`** - FastAPI backend
- Largely AI-generated, adapted from patterns in training notebooks
- Model loading, API endpoints, and caching logic AI-assisted


## External Resources

**Libraries:** Transformers, PEFT, sentence-transformers

**Pre-trained Models:** CLIP ViT-B/32 (OpenAI), Qwen2.5-1.5B-Instruct, bge-base-en-v1.5 (BAAI)

## Datasets

- **Landmark Images**: Scraped from DuckDuckGo Images using custom scripts (`data/images/image_scraper.py`)
- **Wikipedia Context**: Fetched using Wikipedia API (`data/rag/fetch_content.py`)
