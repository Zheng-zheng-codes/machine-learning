import torch
import torch.nn as nn


# =========================
# LSTM / Bi-LSTM 文本分类模型
# =========================
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_classes=2,
        bidirectional=False,
        dropout=0.5,
        pad_idx=0
    ):
        """
        参数说明：
        vocab_size: 词表大小
        embed_dim: 词嵌入维度
        hidden_dim: LSTM 隐藏层维度
        num_layers: LSTM 层数
        num_classes: 分类类别数，IMDB 是二分类，所以为 2
        bidirectional: 是否使用双向 LSTM
        dropout: dropout 比例
        pad_idx: padding 对应的编号，<PAD> = 0
        """

        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 如果是双向 LSTM，方向数为 2；否则为 1
        self.num_directions = 2 if bidirectional else 1

        # =========================
        # 1. 词嵌入层
        # =========================
        # 输入: [batch_size, seq_len]
        # 输出: [batch_size, seq_len, embed_dim]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        # =========================
        # 2. LSTM 层
        # =========================
        # batch_first=True 表示输入输出的第一维是 batch_size
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # =========================
        # 3. Dropout 层
        # =========================
        self.dropout = nn.Dropout(dropout)

        # =========================
        # 4. 全连接分类层
        # =========================
        # 单向 LSTM: hidden_dim
        # 双向 LSTM: hidden_dim * 2
        self.fc = nn.Linear(
            hidden_dim * self.num_directions,
            num_classes
        )

    def forward(self, x):
        """
        前向传播过程。

        x 的形状:
        [batch_size, seq_len]

        返回:
        [batch_size, num_classes]
        """

        # =========================
        # 1. 词嵌入
        # =========================
        # x: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)

        # =========================
        # 2. 输入 LSTM
        # =========================
        # output: 每个时间步的隐藏状态
        # hidden: 最后一个时间步的隐藏状态
        # cell: LSTM 的细胞状态
        output, (hidden, cell) = self.lstm(embedded)

        # =========================
        # 3. 取最终隐藏状态
        # =========================
        if self.bidirectional:
            # 双向 LSTM 的 hidden 形状:
            # [num_layers * 2, batch_size, hidden_dim]
            #
            # hidden[-2] 是最后一层正向 LSTM 的最终隐藏状态
            # hidden[-1] 是最后一层反向 LSTM 的最终隐藏状态
            #
            # 拼接后:
            # [batch_size, hidden_dim * 2]
            final_hidden = torch.cat(
                (hidden[-2], hidden[-1]),
                dim=1
            )
        else:
            # 单向 LSTM 的 hidden 形状:
            # [num_layers, batch_size, hidden_dim]
            #
            # hidden[-1]:
            # [batch_size, hidden_dim]
            final_hidden = hidden[-1]

        # =========================
        # 4. Dropout + 全连接分类
        # =========================
        final_hidden = self.dropout(final_hidden)

        logits = self.fc(final_hidden)

        return logits


# =========================
# 统计模型参数量
# =========================
def count_parameters(model):
    """
    统计模型中需要训练的参数数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# 测试 model.py 是否正常
# =========================
if __name__ == "__main__":
    batch_size = 64
    seq_len = 256
    vocab_size = 20000

    # 随机生成一个 batch 的输入
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long
    )

    # =========================
    # 测试单向 LSTM
    # =========================
    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_classes=2,
        bidirectional=False
    )

    lstm_output = lstm_model(x)

    print("单向 LSTM 输出形状:", lstm_output.shape)
    print("单向 LSTM 参数量:", count_parameters(lstm_model))

    # =========================
    # 测试双向 Bi-LSTM
    # =========================
    bilstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1,
        num_classes=2,
        bidirectional=True
    )

    bilstm_output = bilstm_model(x)

    print("双向 Bi-LSTM 输出形状:", bilstm_output.shape)
    print("双向 Bi-LSTM 参数量:", count_parameters(bilstm_model))