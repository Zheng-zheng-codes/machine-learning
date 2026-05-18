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
    使用空格分词。
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

    注意：
    只能使用训练集构建词表，不能使用测试集，
    否则会造成数据泄露。
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
    将一条文本转换成固定长度的数字序列，并返回真实长度。

    返回：
    ids: padding / 截断后的数字序列，长度固定为 max_len
    length: 文本真实长度，最大不超过 max_len

    例如：
    "this movie is good"
    -> ids = [15, 23, 7, 99, 0, 0, ...]
    -> length = 4
    """
    tokens = tokenize(clean_text(text))

    ids = []
    for token in tokens:
        ids.append(word2idx.get(token, word2idx["<UNK>"]))

    # 真实长度，最多为 max_len
    length = min(len(ids), max_len)

    # 防止极端情况下空文本长度为 0
    if length == 0:
        length = 1

    # 截断
    ids = ids[:max_len]

    # padding
    if len(ids) < max_len:
        ids += [word2idx["<PAD>"]] * (max_len - len(ids))

    return ids, length


# =========================
# 5. 自定义 Dataset
# =========================
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=256):
        """
        texts: 文本数组
        labels: 标签数组
        word2idx: 词表
        max_len: 每条评论保留的最大长度

        这里会提前完成所有文本的编码，
        避免训练过程中反复清洗文本、分词和转编号。
        """
        self.labels = labels
        self.encoded_texts = []
        self.lengths = []

        print("正在提前编码文本数据，请稍等...")

        for text in texts:
            input_ids, length = encode_text(
                text=text,
                word2idx=word2idx,
                max_len=max_len
            )

            self.encoded_texts.append(input_ids)
            self.lengths.append(length)

        print("文本编码完成。")

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, index):
        input_ids = self.encoded_texts[index]
        length = self.lengths[index]
        label = self.labels[index]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        # 注意：
        # 现在返回三个值：
        # input_ids, length, label
        return input_ids, length, label


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
    读取 IMDB 数据集，并返回 train_loader、test_loader、词表大小和词表。

    返回：
    train_loader
    test_loader
    vocab_size
    word2idx
    """

    torch.manual_seed(random_state)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据集文件: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna()

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("CSV 文件中必须包含 review 和 sentiment 两列")

    texts = df["review"].values

    labels = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    }).values

    # 防止标签映射失败
    if pd.isna(labels).any():
        raise ValueError("sentiment 列中存在无法识别的标签，只能是 positive 或 negative")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print("正在构建词表...")

    word2idx = build_vocab(
        texts=train_texts,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq
    )

    print("词表构建完成。")

    train_dataset = IMDBDataset(
        texts=train_texts,
        labels=train_labels,
        word2idx=word2idx,
        max_len=max_len
    )

    test_dataset = IMDBDataset(
        texts=test_texts,
        labels=test_labels,
        word2idx=word2idx,
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
    train_loader, test_loader, vocab_size, word2idx = get_dataloaders(
        csv_path="data/IMDB Dataset.csv",
        batch_size=64,
        max_len=256,
        max_vocab_size=20000,
        min_freq=2,
        test_size=0.2,
        random_state=42
    )

    print("词表大小:", vocab_size)
    print("训练 batch 数量:", len(train_loader))
    print("测试 batch 数量:", len(test_loader))

    for x, lengths, y in train_loader:
        print("输入 x 的形状:", x.shape)
        print("长度 lengths 的形状:", lengths.shape)
        print("标签 y 的形状:", y.shape)
        print("第一条样本:", x[0])
        print("第一条样本真实长度:", lengths[0])
        print("第一条标签:", y[0])
        break