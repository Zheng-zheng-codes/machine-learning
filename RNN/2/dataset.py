import os
import re
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# 0. 自动下载 AG News 数据集
# =========================
def download_agnews_dataset(data_dir="data"):
    """
    如果本地没有 AG News 的 train.csv 和 test.csv，
    则尝试使用 kagglehub 自动下载。
    """

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("已找到 AG News 数据集文件。")
        return

    print("未找到 AG News 数据集，尝试自动下载...")

    os.makedirs(data_dir, exist_ok=True)

    try:
        import kagglehub

        dataset_dir = kagglehub.dataset_download(
            "amananandrai/ag-news-classification-dataset"
        )

        print("数据集下载目录:", dataset_dir)

        found_train = None
        found_test = None

        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                lower_file = file.lower()

                if lower_file == "train.csv":
                    found_train = os.path.join(root, file)

                if lower_file == "test.csv":
                    found_test = os.path.join(root, file)

        if found_train is None or found_test is None:
            raise FileNotFoundError("下载目录中没有找到 train.csv 或 test.csv")

        pd.read_csv(found_train, header=None).to_csv(
            train_path,
            index=False,
            header=False
        )

        pd.read_csv(found_test, header=None).to_csv(
            test_path,
            index=False,
            header=False
        )

        print("AG News 数据集已保存到 data/ 目录。")

    except ImportError:
        print("当前环境未安装 kagglehub。")
        print("请执行：python3 -m pip install kagglehub")
        raise

    except Exception as e:
        print("自动下载 AG News 数据集失败。")
        print("错误信息:", e)
        print("你可以手动下载数据集，并将 train.csv 和 test.csv 放到 data/ 文件夹。")
        raise


# =========================
# 1. 文本清洗函数
# =========================
def clean_text(text):
    """
    对新闻文本进行简单清洗：
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
# 4. 文本编码函数
# =========================
def encode_text(text, word2idx, max_len=128):
    """
    将文本转换为固定长度的数字序列，并返回真实长度。

    返回：
    ids: padding / 截断后的数字序列
    length: 真实文本长度，最大不超过 max_len
    """

    tokens = tokenize(clean_text(text))

    ids = []
    for token in tokens:
        ids.append(word2idx.get(token, word2idx["<UNK>"]))

    length = min(len(ids), max_len)

    # 防止空文本导致 pack_padded_sequence 报错
    if length == 0:
        length = 1

    ids = ids[:max_len]

    if len(ids) < max_len:
        ids += [word2idx["<PAD>"]] * (max_len - len(ids))

    return ids, length


# =========================
# 5. 自定义 AG News Dataset
# =========================
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=128):
        """
        提前完成所有文本编码，避免训练过程中重复清洗和分词。
        """

        self.labels = labels
        self.encoded_texts = []
        self.lengths = []

        print("正在提前编码 AG News 文本数据，请稍等...")

        for text in texts:
            input_ids, length = encode_text(
                text=text,
                word2idx=word2idx,
                max_len=max_len
            )

            self.encoded_texts.append(input_ids)
            self.lengths.append(length)

        print("AG News 文本编码完成。")

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, index):
        input_ids = self.encoded_texts[index]
        length = self.lengths[index]
        label = self.labels[index]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, length, label


# =========================
# 6. 读取 AG News CSV
# =========================
def read_agnews_csv(csv_path):
    """
    读取 AG News 的 train.csv 或 test.csv。

    兼容两种格式：
    1. 没有表头：
       1,title,description

    2. 有表头：
       Class Index,Title,Description
       1,title,description

    标签：
    1 -> World
    2 -> Sports
    3 -> Business
    4 -> Sci/Tech

    训练时转换为：
    0, 1, 2, 3
    """

    df = pd.read_csv(
        csv_path,
        header=None,
        names=["label", "title", "description"]
    )

    # 如果第一行是表头，就删掉
    if str(df.iloc[0]["label"]).lower() in ["class index", "classindex", "label"]:
        df = df.iloc[1:].reset_index(drop=True)

    df = df.dropna()

    texts = (
        df["title"].astype(str)
        + " "
        + df["description"].astype(str)
    ).values

    labels = df["label"].astype(int).values - 1

    return texts, labels


# =========================
# 7. 获取 DataLoader
# =========================
def get_dataloaders(
    data_dir="data",
    batch_size=64,
    max_len=128,
    max_vocab_size=20000,
    min_freq=2,
    random_state=42
):
    """
    返回 AG News 的训练集和测试集 DataLoader。
    """

    torch.manual_seed(random_state)

    download_agnews_dataset(data_dir)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集文件: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集文件: {test_path}")

    train_texts, train_labels = read_agnews_csv(train_path)
    test_texts, test_labels = read_agnews_csv(test_path)

    print("正在构建词表...")

    word2idx = build_vocab(
        texts=train_texts,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq
    )

    print("词表构建完成。")

    train_dataset = AGNewsDataset(
        texts=train_texts,
        labels=train_labels,
        word2idx=word2idx,
        max_len=max_len
    )

    test_dataset = AGNewsDataset(
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
# 8. 单独测试 dataset.py
# =========================
if __name__ == "__main__":
    train_loader, test_loader, vocab_size, word2idx = get_dataloaders()

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