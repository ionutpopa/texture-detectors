import os
import shutil

# Path to your training data folders
root_dir = 'fabrics_dataset/fabric/train'

# Loop through all subfolders (fabric classes)
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if len(image_files) < 930:
            print(f"Deleting '{folder_name}' with only {len(image_files)} images.")
            shutil.rmtree(folder_path)
