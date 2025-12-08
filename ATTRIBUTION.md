# Attribution

## AI-Generated Code

### Data Collection (`data/`)

**`data/images/image_scraper.py`** - Image scraping script
- AI-assisted, particularly `download_images_for_landmark` function
- DuckDuckGo Images scraping syntax and stock image filtering logic generated with AI

### Training Notebooks (`notebooks/`)

**`notebooks/clip_classifier.ipynb`** - Classifier training notebook
- Data augmentation pipeline (`train_augment` transforms composition) AI-assisted
- Training loop boilerplate functions AI-assisted
- Plotting code and results formatting AI-assisted

### Web Application (`src/`)

**`src/index.html`** - Frontend interface
- Almost entirely AI-generated through iterative prompting

**`src/app.py`** - FastAPI backend
- Largely AI-generated, adapted from patterns in training notebooks
- Model loading, API endpoints, and caching logic AI-assisted


## External Resources

**Libraries:** Transformers, PEFT, sentence-transformers

**Pre-trained Models:** CLIP ViT-B/32 (OpenAI), Qwen2.5-3B-Instruct, all-MiniLM-L6-v2 (sentence-transformers)

## Datasets

- **Landmark Images**: Scraped from DuckDuckGo Images using custom scripts (`data/images/image_scraper.py`)
- **Wikipedia Context**: Fetched using Wikipedia API (`data/rag/fetch_content.py`)