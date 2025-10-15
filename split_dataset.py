import os
import random
import shutil

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8):
    images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    for split_name, split_list in [("train", train_images), ("val", val_images)]:
        image_output = os.path.join(output_dir, "images", split_name)
        label_output = os.path.join(output_dir, "labels", split_name)
        os.makedirs(image_output, exist_ok=True)
        os.makedirs(label_output, exist_ok=True)

        for img_file in split_list:
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"

            # Copy image
            shutil.copy(os.path.join(image_dir, img_file), os.path.join(image_output, img_file))
            
            # Copy label
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(label_output, label_file))
            else:
                print(f"Warning: Label not found for image {img_file}")

if __name__ == "__main__":
    split_dataset(
        image_dir="data/images",
        label_dir="data/labels",
        output_dir="data",
        train_ratio=0.8
    )
