# Setup Instructions

## Prerequisites

- Python 3.10 or higher
- ~8GB RAM (for loading the transformer model)
- ~15GB disk space (for models and data)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amoghmanral/landmarks.git
   cd landmarks
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

The landmark images are not included in the repository due to size. To set up the data:

1. Download the landmark images zip from the provided Box link: https://duke.box.com/s/w72a997xljs1haztpb3c9fzjr523fmb7
2. Unzip the file
3. Place the contents into:
   ```
   data/images/landmark_images/
   ```

The final structure should look like:
```
data/
├── images/
│   └── landmark_images/
│       ├── 0001_Eiffel_Tower/
│       │   ├── 01.jpg
│       │   ├── 02.jpg
│       │   └── ...
│       ├── 0002_Colosseum/
│       └── ...
├── train.csv
├── val.csv
└── test.csv
```

## Running the Application

```bash
python src/app.py
```

Then open http://localhost:8000 in your browser.

## First Run

On the first run, the application will automatically download the Qwen2.5-1.5B-Instruct model (~3GB).

All other models (CLIP, LoRA weights, classifier) are included in the repository.