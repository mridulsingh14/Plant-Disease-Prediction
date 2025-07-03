import os
import shutil
import random

# Path to your PlantVillage dataset
DATASET_DIR = 'PlantVillage'

# Get class folder names and image counts
class_counts = {}
for class_dir in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_dir)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[class_dir] = count

max_count = max(class_counts.values())

for class_dir, count in class_counts.items():
    class_path = os.path.join(DATASET_DIR, class_dir)
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    i = 0
    while len(images) < max_count:
        img_to_copy = random.choice(images)
        src = os.path.join(class_path, img_to_copy)
        dst = os.path.join(class_path, f"aug_{i}_{img_to_copy}")
        shutil.copy(src, dst)
        images.append(f"aug_{i}_{img_to_copy}")
        i += 1
    print(f"{class_dir} now has {len(images)} images.")

print("Dataset balancing complete!") 