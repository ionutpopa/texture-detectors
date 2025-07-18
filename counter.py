import json
from collections import Counter
import pandas as pd

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def count_labels(annotations, key):
    return Counter([item[key] for item in annotations])

def display_counts(title, counter):
    print(f"\n📊 {title}")
    df = pd.DataFrame(counter.items(), columns=[title, "count"]).sort_values(by="count", ascending=False)
    print(df.to_string(index=False))

def main():
    json_path = "fabric_clip_annotations.json"
    annotations = load_annotations(json_path)
    
    total_images = len(annotations)
    print(f"Total images annotated: {total_images}")

    fabric_counts = count_labels(annotations, "fabric_class")
    weight_counts = count_labels(annotations, "material_weight")
    finish_counts = count_labels(annotations, "finish")

    display_counts("fabric_class", fabric_counts)
    display_counts("material_weight", weight_counts)
    display_counts("finish", finish_counts)

if __name__ == "__main__":
    main()
