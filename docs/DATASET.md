# Custom Dataset Generation

This documents the custom dataset creation process.

## Overview

The dataset consists of three components:
1. **Images**: ~37,500 images across 938 landmarks (40 per landmark)
2. **Text Descriptions**: GPT-generated descriptions for contrastive learning
3. **Wikipedia Content**: Full article text for RAG-based Q&A

## 1. Image Collection

**Script**: `data/images/image_scraper.py`

### Process
- Source: DuckDuckGo image search API
- Target: 40 images per landmark
- Total: 938 landmarks × 40 images = ~37,520 images

### Filtering
Attempted to exclude stock photo sites:
- Shutterstock, iStockPhoto, Getty Images, Adobe Stock, etc.


### Validation
Each downloaded image is verified using PIL to ensure it's a valid, non-corrupted image file.


## 2. Text Description Generation

**Script**: `data/clip/fetch_descriptions.py`

### Process
- Model: GPT-4o-mini via OpenAI API
- Batch size: 5 landmarks per API call
- Descriptions per landmark: 22 total (10 train, 6 val, 6 test)

### Description Styles

**Training Descriptions (10 per landmark)**
- Dense, encyclopedic, informative
- Include: era, architectural style, cultural context, builder/dynasty
- Focus varies: historical, visual, cultural, setting, mixed perspectives

**Validation Descriptions (6 per landmark)**
- Semi-formal, moderately detailed
- Mix of precise and slightly casual language
- Tests generalization to unseen phrasings

**Test Descriptions (6 per landmark)**
- Casual, conversational, not too specific
- Simulates real user search queries

### Critical Rule
Descriptions never contain the landmark name itself. This forces the model to learn visual/semantic associations rather than memorizing name-to-name mappings.


## 3. Wikipedia Content for RAG

**Scripts**:
- `data/rag/fetch_wiki_url.py` - Find Wikipedia URLs
- `data/rag/fetch_content.py` - Extract article content

### URL Resolution
1. Try direct Wikipedia API lookup with landmark name - direct method
2. If not found, use Wikipedia search API - search method
3. Manual checks and replacements for the search method

### Content Extraction
- Uses Wikipedia API with `explaintext` for plain text
- Stores full article content for each landmark