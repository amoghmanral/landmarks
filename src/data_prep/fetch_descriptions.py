import openai
import pandas as pd
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

BATCH_SIZE = 5


def generate_descriptions_batch(landmarks_batch):
    
    landmarks_text = ""
    for idx, (name, country) in enumerate(landmarks_batch, 1):
        landmarks_text += f"{idx}. {name}\n"
    
    prompt = f"""Generate 22 diverse descriptions for EACH landmark - 10 for training, 6 for validation, and 6 for testing in CLIP fine-tuning.

CRITICAL RULES:
1. Use the EXACT landmark names provided below as JSON keys
2. DO NOT include the landmark name in any description
3. Describe what you see/know WITHOUT naming it

Landmarks:
{landmarks_text}

TRAINING DESCRIPTIONS (10 total):
Purpose: Teach the model core visual and historical features
Style:
- Dense, informative, encyclopedic
- 1-2 sentences maximum
- Include: era/period, architectural style, cultural context, builder/dynasty, location/country
- Focus on different aspects per description:
  * Historical era and architectural style (2 descriptions)
  * Visual features and materials (2 descriptions)
  * Cultural/religious significance (2 descriptions)
  * Setting, location, and landscape (2 descriptions)
  * Mixed perspectives and details (2 descriptions)
- Vary sentence structure and vocabulary across all 10
- Describe it WITHOUT naming it
- You can mention the country/location, just not the landmark name

VALIDATION DESCRIPTIONS (6 total):
Purpose: Test generalization to descriptions the model hasn't seen
Style:
- Semi-formal, informative but less encyclopedic than training
- Mix of precise and slightly casual language
- Include some information from training but phrased differently
- 1-2 sentences, balanced tone
- Don't always include all details (era, location, style)
- Describe it WITHOUT naming it
- You can mention the country/location

TEST DESCRIPTIONS (6 total):
Purpose: Simulate real user queries trying to identify what they saw
Style:
- Casual, conversational, tourist-like
- How a person might describe or search for it
- Slightly vague, subjective impressions mixed with facts
- 1-2 sentences, natural language
- Describe it WITHOUT naming it
- You can mention the country/location
- Examples: "That famous tower with the clock in London", "Beautiful white building with domes in India"

CRITICAL: Output ONLY valid JSON. Do not include any markdown formatting, code blocks, or explanatory text.
CRITICAL: NEVER include the landmark name in descriptions. Describe features, location, history - but not the name itself.

Format as JSON with exact landmark names provided as keys:
{{
  "Exact Landmark Name": {{
    "train": [
      "Description 1",
      "Description 2",
      ...exactly 10 descriptions total
    ],
    "validation": [
      "Description 1",
      "Description 2",
      ...exactly 6 descriptions total
    ],
    "test": [
      "Description 1",
      "Description 2",
      ...exactly 6 descriptions total
    ]
  }},
  ...
}}

Example for Taj Mahal (note: name NOT included in descriptions, but country IS):
{{
  "Taj Mahal": {{
    "train": [
      "17th century Mughal mausoleum in Agra, India, featuring white marble construction and large central dome.",
      "Iconic symmetrical monument with four minarets and Persian-influenced gardens with reflecting pools.",
      "UNESCO World Heritage Site showcasing Indo-Islamic architecture with detailed floral patterns in marble.",
      "Massive white marble tomb complex in India commissioned by Shah Jahan for his wife Mumtaz Mahal.",
      "Monumental structure featuring intricate pietra dura inlay work with semi-precious stones.",
      "Perfectly symmetrical design with octagonal central chamber and four identical facades.",
      "Symbol of Mughal architectural achievement combining Persian, Islamic, and Indian design elements.",
      "Sacred Islamic mausoleum complex with mosque and guest house flanking the main tomb.",
      "Riverfront monument on the Yamuna River in India, set within 42-acre gardens with Mughal char bagh layout.",
      "Agra landmark visible for miles, situated on elevated marble platform with corner minarets."
    ],
    "validation": [
      "Famous white marble mausoleum in India, known for its perfect symmetry and massive central dome.",
      "Historic Mughal monument featuring intricate decorative patterns and surrounded by formal gardens.",
      "Iconic building in Agra with four tall minarets and detailed stonework throughout.",
      "UNESCO site showcasing Indo-Islamic architecture with reflecting pools and symmetrical design.",
      "17th century tomb complex built by Mughal emperor, featuring elaborate marble inlay work.",
      "Monumental structure on the Yamuna River, famous for changing colors at different times of day."
    ],
    "test": [
      "That famous white building in India, really beautiful with a big dome",
      "Impressive marble monument with towers at the corners and gorgeous gardens",
      "Where's that place in India that looks amazing at sunrise?",
      "Beautiful old structure with detailed decorations, very symmetrical design",
      "That iconic Indian landmark with the reflecting pools in front",
      "Huge white building with intricate patterns, one of the most photographed places"
    ]
  }}
}}

IMPORTANT: Ensure clear distinction between sets:
- Train: Most detailed and varied, encyclopedic, NO NAME but country OK
- Validation: Moderate detail, natural but informative, NO NAME but country OK
- Test: Casual, conversational, user-query-like, NO NAME but country OK

Generate the JSON:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that generates diverse landmark descriptions for CLIP fine-tuning. NEVER include the landmark name in descriptions. Create clear distinctions between training (encyclopedic), validation (semi-formal), and test (casual) descriptions. Output valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=5000,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  Error: {e}")
        return None


def validate_descriptions(data):
    if not isinstance(data, dict):
        return False
    return (
        len(data.get('train', [])) == 10 and
        len(data.get('validation', [])) == 6 and
        len(data.get('test', [])) == 6
    )


def main():

    train_df = pd.read_csv("../../data/train.csv")
    val_df = pd.read_csv("../../data/train.csv")
    test_df = pd.read_csv("../../data/train.csv")
    list_df = pd.read_csv("landmark_list.csv")
    
    list_df['Name'] = list_df['Name'].str.replace('\xa0', ' ')
    landmarks = list(zip(list_df['Name'], list_df['Country']))
        
    descriptions = {}
    for i in range(0, len(landmarks), BATCH_SIZE):
        batch = landmarks[i:i + BATCH_SIZE]
        print(f"Batch {i // BATCH_SIZE + 1}")   
             
        result = generate_descriptions_batch(batch)
        if result:
            for name, _ in batch:
                data = result.get(name)
                if data and validate_descriptions(data):
                    descriptions[name] = data
                else:
                    print(f"Failed: {name}")
        
        time.sleep(1)
    
    with open("descriptions.json", "w") as f:
        json.dump({"descriptions": descriptions}, f, indent=2)
    print(f"\nSaved {len(descriptions)} landmarks to descriptions.json")
    
    def pair(df, key):
        rows = []
        for _, row in df.iterrows():
            if row['landmark_name'] in descriptions:
                for desc in descriptions[row['landmark_name']][key]:
                    rows.append({**row, 'description': desc})
        return pd.DataFrame(rows)
    
    pair(train_df, "train").to_csv("train.csv", index=False)
    pair(val_df, "validation").to_csv("val.csv", index=False)
    pair(test_df, "test").to_csv("test.csv", index=False)
    

if __name__ == "__main__":
    main()