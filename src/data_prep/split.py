import pandas as pd
import random

random.seed(42)

df = pd.read_csv('landmark_list.csv')
df['Name'] = df['Name'].str.replace('\xa0', ' ')

train_data = []
val_data = []
test_data = []

for idx, row in df.iterrows():
    name = row['Name']
    folder = f"{str(idx + 1).zfill(4)}_{name.replace(' ', '_').replace('/', '_')}"
    
    images = list(range(1, 41))
    random.shuffle(images)
    
    for img_num in images[:28]:
        train_data.append({
            'image_path': f"landmark_images/{folder}/{img_num}.jpg",
            'landmark_name': name,
            'landmark_idx': idx
        })
    
    for img_num in images[28:34]:
        val_data.append({
            'image_path': f"landmark_images/{folder}/{img_num}.jpg",
            'landmark_name': name,
            'landmark_idx': idx
        })
    
    for img_num in images[34:40]:
        test_data.append({
            'image_path': f"landmark_images/{folder}/{img_num}.jpg",
            'landmark_name': name,
            'landmark_idx': idx
        })

pd.DataFrame(train_data).to_csv('../../data/train.csv', index=False)
pd.DataFrame(val_data).to_csv('../../data/val.csv', index=False)
pd.DataFrame(test_data).to_csv('../../data/test.csv', index=False)