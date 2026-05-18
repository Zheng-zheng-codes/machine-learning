import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


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
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        # =========================
        # 2. LSTM 层
        # =========================
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
        self.fc = nn.Linear(
            hidden_dim * self.num_directions,
            num_classes
        )

    def forward(self, x, lengths):
        """
        前向传播过程。

        参数：
        x: [batch_size, seq_len]
        lengths: [batch_size]，每条文本真实长度，不包含 padding

        返回：
        logits: [batch_size, num_classes]
        """

        # =========================
        # 1. 词嵌入
        # =========================
        # x: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)

        # =========================
        # 2. 打包序列，让 LSTM 忽略 padding
        # =========================
        packed_embedded = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # =========================
        # 3. 输入 LSTM
        # =========================
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # =========================
        # 4. 取最终隐藏状态
        # =========================
        if self.bidirectional:
            # 双向 LSTM：
            # hidden[-2] 是最后一层正向 LSTM 的最终隐藏状态
            # hidden[-1] 是最后一层反向 LSTM 的最终隐藏状态
            final_hidden = torch.cat(
                (hidden[-2], hidden[-1]),
                dim=1
            )
        else:
            # 单向 LSTM：
            # hidden[-1] 是最后一层的最终隐藏状态
            final_hidden = hidden[-1]

        # =========================
        # 5. Dropout + 全连接分类
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

    # 随机生成真实长度，范围为 1 到 seq_len
    lengths = torch.randint(
        low=1,
        high=seq_len + 1,
        size=(batch_size,),
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

    lstm_output = lstm_model(x, lengths)

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

    bilstm_output = bilstm_model(x, lengths)

    print("双向 Bi-LSTM 输出形状:", bilstm_output.shape)
    print("双向 Bi-LSTM 参数量:", count_parameters(bilstm_model))