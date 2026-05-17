import torch
import torch.nn as nn


# =========================
# LSTM 新闻分类模型
# =========================
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_classes=4,
        dropout=0.5,
        pad_idx=0
    ):
        """
        LSTM 文本分类模型。

        参数：
        vocab_size: 词表大小
        embed_dim: 词嵌入维度
        hidden_dim: 隐藏层维度
        num_layers: LSTM 层数
        num_classes: AG News 是 4 分类
        dropout: dropout 比例
        pad_idx: padding 对应编号
        """

        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """

        embedded = self.embedding(x)
        # embedded: [batch_size, seq_len, embed_dim]

        output, (hidden, cell) = self.lstm(embedded)

        # hidden: [num_layers, batch_size, hidden_dim]
        final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)

        logits = self.fc(final_hidden)

        return logits


# =========================
# GRU 新闻分类模型
# =========================
class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_classes=4,
        dropout=0.5,
        pad_idx=0
    ):
        """
        GRU 文本分类模型。

        GRU 只有隐藏状态，没有 LSTM 中独立的 cell state。
        """

        super(GRUClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """

        embedded = self.embedding(x)
        # embedded: [batch_size, seq_len, embed_dim]

        output, hidden = self.gru(embedded)

        # hidden: [num_layers, batch_size, hidden_dim]
        final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)

        logits = self.fc(final_hidden)

        return logits


# =========================
# 统计模型参数量
# =========================
def count_parameters(model):
    """
    统计模型中可训练参数数量。
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# 单独测试 model.py
# =========================
if __name__ == "__main__":
    batch_size = 64
    seq_len = 128
    vocab_size = 20000
    num_classes = 4

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long
    )

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_classes=num_classes
    )

    gru_model = GRUClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_classes=num_classes
    )

    lstm_output = lstm_model(x)
    gru_output = gru_model(x)

    print("LSTM 输出形状:", lstm_output.shape)
    print("LSTM 参数量:", count_parameters(lstm_model))

    print("GRU 输出形状:", gru_output.shape)
    print("GRU 参数量:", count_parameters(gru_model))