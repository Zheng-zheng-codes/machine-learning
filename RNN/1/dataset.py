import os
import re
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# 1. 文本清洗函数
# =========================
def clean_text(text):
    """
    对电影评论文本进行简单清洗：
    1. 转小写
    2. 去掉 HTML 标签
    3. 去掉非字母字符
    4. 合并多余空格
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# 2. 分词函数
# =========================
def tokenize(text):
    """
    使用最简单的空格分词。
    例如：
    "this movie is good" -> ["this", "movie", "is", "good"]
    """
    return text.split()


# =========================
# 3. 构建词表
# =========================
def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    """
    根据训练集文本构建词表。
    """
    counter = Counter()

    for text in texts:
        tokens = tokenize(clean_text(text))
        counter.update(tokens)

    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, freq in counter.most_common(max_vocab_size - 2):
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    return word2idx


# =========================
# 4. 文本转数字序列
# =========================
def encode_text(text, word2idx, max_len=256):
    """
    将一条文本转换成固定长度的数字序列。
    """
    tokens = tokenize(clean_text(text))

    ids = []
    for token in tokens:
        ids.append(word2idx.get(token, word2idx["<UNK>"]))

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [word2idx["<PAD>"]] * (max_len - len(ids))

    return ids


# =========================
# 5. 自定义 Dataset
# =========================
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=256):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        input_ids = encode_text(text, self.word2idx, self.max_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, label


# =========================
# 6. 获取 DataLoader
# =========================
def get_dataloaders(
    csv_path="data/IMDB Dataset.csv",
    batch_size=64,
    max_len=256,
    max_vocab_size=20000,
    min_freq=2,
    test_size=0.2,
    random_state=42
):
    """
    读取 IMDB 数据集，并返回 train_loader、test_loader、词表大小。
    """

    torch.manual_seed(random_state)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据集文件: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna()

    texts = df["review"].values
    labels = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    }).values

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    word2idx = build_vocab(
        train_texts,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq
    )

    train_dataset = IMDBDataset(
        train_texts,
        train_labels,
        word2idx,
        max_len=max_len
    )

    test_dataset = IMDBDataset(
        test_texts,
        test_labels,
        word2idx,
        max_len=max_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    vocab_size = len(word2idx)

    return train_loader, test_loader, vocab_size, word2idx


# =========================
# 7. 测试 dataset.py 是否正常
# =========================
if __name__ == "__main__":
    train_loader, test_loader, vocab_size, word2idx = get_dataloaders()

    print("词表大小:", vocab_size)
    print("训练 batch 数量:", len(train_loader))
    print("测试 batch 数量:", len(test_loader))

    for x, y in train_loader:
        print("输入 x 的形状:", x.shape)
        print("标签 y 的形状:", y.shape)
        print("第一条样本:", x[0])
        print("第一条标签:", y[0])
        break