import os
import shutil

val_dir = './data/tiny-imagenet-200/val'
img_dir = os.path.join(val_dir, 'images')

anno_file = os.path.join(val_dir, 'val_annotations.txt')

with open(anno_file, 'r') as f:
    data = f.readlines()

for line in data:
    words = line.split('\t')

    img_name = words[0]
    folder = words[1]

    folder_path = os.path.join(val_dir, folder)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    src = os.path.join(img_dir, img_name)
    dst = os.path.join(folder_path, img_name)

    shutil.move(src, dst)

print("TinyImageNet验证集整理完成")