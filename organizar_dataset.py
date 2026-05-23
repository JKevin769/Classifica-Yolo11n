import os
import shutil
import random

source_dir = "/home/vicom04/yolo11/imagens"
base_dir = "/home/vicom04/yolo11/datasets"

train_ratio = 0.8

for classe in os.listdir(source_dir):
    classe_path = os.path.join(source_dir, classe)

    if not os.path.isdir(classe_path):
        continue

    imagens = [
        f for f in os.listdir(classe_path)
        if os.path.isfile(os.path.join(classe_path, f))
    ]

    random.shuffle(imagens)

    split = int(len(imagens) * train_ratio)
    train_imgs = imagens[:split]
    val_imgs = imagens[split:]

    for img in train_imgs:
        src = os.path.join(classe_path, img)
        dst = os.path.join(base_dir, "train", classe, img)
        shutil.copy(src, dst)

    for img in val_imgs:
        src = os.path.join(classe_path, img)
        dst = os.path.join(base_dir, "val", classe, img)
        shutil.copy(src, dst)

print("Dataset organizado com sucesso!")
