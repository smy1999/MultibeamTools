import pandas as pd
from ConversionToolbox import AmplitudeGrazingAngle as AGA
from CurveFittingToolbox import GSAB as cf
import numpy as np
import math
from matplotlib import pyplot as plt
import time


def test_rdxls():
    path = 'C:/Users/86598/PycharmProjects/MultibeamTools/'
    file_name = '0158_20160830_025854_EX1607_MB.all.mb58.xlsx'
    file = path + file_name
    data = pd.read_excel(file, sheet_name='Sheet1', engine='openpyxl')
    print(data)
    print(type(data))


def test_new_curve():
    from ConversionToolbox import AmplitudeGrazingAngle as aga
    from CurveFittingToolbox import GSAB as cf
    import pandas as pd
    file_path = 'D:/Study/mbsystem/mbbackangle/compare/0158_20160830_025854_EX1607_MB_slope.all.mb58_tot.aga'
    table = aga.AGATable(file_path)
    xls_path = '../'
    xls_name = table.write_xls(xls_path)
    df = pd.read_excel(xls_name, engine='openpyxl')
    x = df['angle'].values
    y = df['beam amplitude'].values

    x = cf.xconvert(x, 'degree')
    x, y = cf.get_interval(x, y)
    cf.fitGSAB(x, y, if_save_img=True)


def test_try(type):
    start = time.time()
    a = ['0', '0.12', '5.0', '5.1', '5.12.015']
    for i in range(10):
        a = a + a

    if type:
        for index, i in enumerate(a):
            try:
                a[index] = float(i)
                try:
                    a[index] = int(i)
                except ValueError:
                    pass
            except ValueError:
                pass
    else:
        for index in range(len(a)):
            try:
                a[index] = float(a[index])
                try:
                    a[index] = int(a[index])
                except ValueError:
                    pass
            except ValueError:
                pass
    end = time.time()
    return end - start


def test_time():
    time1 = 0
    time2 = 0
    for i in range(100):
        time1 += test_try(True)
        time2 += test_try(False)
    print(time1)
    print(time2)


def test_softmax():
    x = [12, 14]
    print(softmax(x))


def softmax(x):
    y = np.power(np.e, x)
    s = y.sum()
    y = y / y.sum()
    return y


def test_pe():
    print(test_position_encoding(4, 0))
    print(test_position_encoding(4, 1))
    print(test_position_encoding(4, 2))
    print(test_position_encoding(4, 3))


def test_position_encoding(dimension, t):
    pos = []
    for k in range(int(dimension / 2)):
        w = 1 / np.power(10000, 2 * k / dimension)
        pos.append(np.sin(w * t))
        pos.append(np.cos(w * t))
    return pos


def test_binary():
    n = 10
    p = 0.57
    ans = []
    for i in range(n + 1):
        ans.append(binary(n, p, i))
    plt.bar(range(n + 1), ans)
    plt.show()


def binary(n, p, k):
    coefficient = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    ans = coefficient * math.pow(p, k) * math.pow(1 - p, n - k)
    return ans


def test_transformer_position_encoding():
    import seaborn as sns
    import math

    def get_positional_encoding(max_seq_len, embed_dim):
        # 初始化一个positional encoding
        # embed_dim: 字嵌入的维度
        # max_seq_len: 最大的序列长度
        positional_encoding = \
            np.array(
                [
            [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
            if pos != 0 else np.zeros(embed_dim)
                    for pos in range(max_seq_len)
                ]
            )

        # print(positional_encoding)
        # print('max_seq_len =', max_seq_len)
        # print('embed_dim =', embed_dim)
        # print(positional_encoding.shape)

        positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数
        positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
        # 归一化, 用位置嵌入的每一行除以它的模长
        # denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        # position_enc = position_enc / (denominator + 1e-8)
        return positional_encoding

    positional_encoding = get_positional_encoding(max_seq_len=100, embed_dim=16)
    plt.figure(figsize=(10, 10))
    sns.heatmap(positional_encoding)
    plt.title("Sinusoidal Function")
    plt.xlabel("hidden dimension")
    plt.ylabel("sequence length")
    plt.show()


if __name__ == '__main__':
    test_transformer_position_encoding()



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # self.temperature是论文中的d_k ** 0.5，防止梯度过大
        # QxK/sqrt(dk)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 屏蔽不想要的输出
            attn = attn.masked_fill(mask == 0, -1e9)
        # softmax+dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 概率分布xV
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    # n_head头的个数，默认是8
    # d_model编码向量长度，例如本文说的512
    # d_k, d_v的值一般会设置为 n_head * d_k=d_model，
    # 此时concat后正好和原始输入一样，当然不相同也可以，因为后面有fc层
    # 相当于将可学习矩阵分成独立的n_head份
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        # 假设n_head=8，d_k=64
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # d_model输入向量，n_head * d_k输出向量
        # 可学习W^Q，W^K,W^V矩阵参数初始化
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # 最后的输出维度变换操作
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        # 单头自注意力
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # 假设qkv输入是(b,100,512),100是训练每个样本最大单词个数
        # 一般qkv相等，即自注意力
        residual = q
        # 将输入x和可学习矩阵相乘，得到(b,100,512)输出
        # 其中512的含义其实是8x64，8个head，每个head的可学习矩阵为64维度
        # q的输出是(b,100,8,64),kv也是一样
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # 变成(b,8,100,64)，方便后面计算，也就是8个头单独计算
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        # 输出q是(b,8,100,64),维持不变,内部计算流程是：
        # q*k转置，除以d_k ** 0.5，输出维度是b,8,100,100即单词和单词直接的相似性
        # 对最后一个维度进行softmax操作得到b,8,100,100
        # 最后乘上V，得到b,8,100,64输出
        q, attn = self.attention(q, k, v, mask=mask)

        # b,100,8,64-->b,100,512
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # 残差计算
        q += residual
        # 层归一化，在512维度计算均值和方差，进行层归一化
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 两个fc层，对最后的512维度进行变换
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x