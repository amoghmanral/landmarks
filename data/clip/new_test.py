import openai
import pandas as pd
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

BATCH_SIZE = 10


def generate_descriptions_batch(landmarks_batch):
    
    landmarks_text = ""
    for idx, (name, country) in enumerate(landmarks_batch, 1):
        landmarks_text += f"{idx}. {name} ({country})\n"
    
    prompt = f"""Generate exactly 2 test descriptions for EACH landmark below.

PURPOSE: These descriptions will query a CLIP model that suggests the TOP 5 matching landmarks. The goal is for the correct landmark to appear in those top 5 matches alongside similar landmarks.

CRITICAL RULES:
1. Use the EXACT landmark names provided as JSON keys
2. DO NOT include the landmark name and exact location in any description
3. Strike a balance between vague and specific

Landmarks:
{landmarks_text}

DESCRIPTION APPROACH:
Descriptions should use CATEGORICAL features that define a CLASS of similar landmarks, while including some detail to boost the correct match into the top 5.

Include 2-3 of these categorical features:
- Architectural style category (e.g., "Gothic cathedral", "Mughal mausoleum", "ancient Greek temple", "Art Deco skyscraper")
- Material type (e.g., "white marble", "limestone", "red brick", "steel and glass")
- Structural category (e.g., "domed building", "spired church", "tiered pagoda", "arched bridge")
- Era/period (e.g., "medieval", "17th century", "ancient", "modern")
- Setting type (e.g., "hilltop fortress", "riverside", "city square", "coastal cliff")
- Function category (e.g., "royal palace", "religious temple", "memorial monument", "defensive castle")

BALANCE EXAMPLES:

TOO VAGUE (matches too many):
"Old building in Europe"
"historic monument"


TOO SPECIFIC (only matches one):
"Building with 324 meters height and iron lattice structure built for 1889 World's Fair" (Eiffer Tower)
"White marble Mughalian building in India with with large central dome and reflecting pools" (Taj Mahal)


GOOD examples (specific, visual, searchable):
- "white marble mausoleum with reflecting pool"
- "colorful row houses on a steep hill"
- "ancient amphitheater overlooking the Mediterranean"
- "red torii gates in a forest path"
- "gothic cathedral with twin spires"


STYLE:
- natural phrases
- Descriptive but not encyclopedic
- Like someone describing visual features what they remember seeing


CRITICAL: Output ONLY valid JSON. No markdown, no code blocks.
CRITICAL: Do not include landmark names and exact locations in descriptions.
CRITICAL: Remember the objective - top k matching. Design with the aim to get the landmark into the TOP 3-5 matches.

Format:
{{
  "Exact Landmark Name": {{
    "test": [
      "Description 1 with categorical features",
      "Description 2 with different categorical features"
    ]
  }},
  ...
}}

Generate the JSON:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that generates landmark descriptions for CLIP querying. Create descriptions using categorical architectural/historical/setting features that match a CLASS of similar landmarks while boosting the correct one into top results. NEVER include the landmark name. Output valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  Error: {e}")
        return None


def validate_descriptions(data):
    if not isinstance(data, dict):
        return False
    return len(data.get('test', [])) == 2


def main():
    list_df = pd.read_csv("landmark_list.csv")
    
    list_df['Name'] = list_df['Name'].str.replace('\xa0', ' ')
    landmarks = list(zip(list_df['Name'], list_df['Country']))
        
    descriptions = {}
    failed = []
    
    for i in range(0, len(landmarks), BATCH_SIZE):
        batch = landmarks[i:i + BATCH_SIZE]
        print(f"Batch {i // BATCH_SIZE + 1}/{(len(landmarks) + BATCH_SIZE - 1) // BATCH_SIZE}")   
             
        result = generate_descriptions_batch(batch)
        if result:
            for name, country in batch:
                data = result.get(name)
                if data and validate_descriptions(data):
                    descriptions[name] = data
                    print(f"  ✓ {name}")
                else:
                    failed.append({"name": name, "country": country, "reason": "invalid or missing data"})
                    print(f"  ✗ Failed: {name}")
        else:
            # Entire batch failed
            for name, country in batch:
                failed.append({"name": name, "country": country, "reason": "batch API error"})
                print(f"  ✗ Failed: {name}")
        
        time.sleep(1)
    
    with open("test_descriptions.json", "w") as f:
        json.dump({"descriptions": descriptions}, f, indent=2)
    print(f"\nSaved {len(descriptions)} landmarks to test_descriptions.json")
    
    if failed:
        with open("failed_landmarks.json", "w") as f:
            json.dump({"failed": failed, "count": len(failed)}, f, indent=2)
        print(f"Saved {len(failed)} failed landmarks to failed_landmarks.json")
    
    # Create test CSV
    test_df = pd.read_csv("/final_data/test.csv")
    
    rows = []
    for _, row in test_df.iterrows():
        if row['landmark_name'] in descriptions:
            for desc in descriptions[row['landmark_name']]['test']:
                rows.append({**row, 'description': desc})
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv("/final_data/new_test.csv", index=False)
    print(f"Created test CSV with {len(result_df)} rows")


if __name__ == "__main__":
    main()