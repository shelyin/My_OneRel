import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from transformers import BertModel  # 导入Hugging Face的BertModel，用于加载预训练的BERT模型

class OneRel(nn.Module):
    def __init__(self, config):
        """
        初始化OneRel模型
        :param config: 配置参数（如bert模型路径、BERT维度、关系数、标签大小等）
        """
        super(OneRel, self).__init__()  # 初始化父类nn.Module
        self.config = config  # 保存配置参数

        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        
        # 定义一个全连接层，将BERT的输出映射到关系和标签的空间
        self.relation_linear = nn.Linear(self.config.bert_dim * 3, self.config.num_rel * self.config.tag_size)
        
        # 定义一个全连接层，用于将实体对的表示映射到一个新的空间
        self.project_matrix = nn.Linear(self.config.bert_dim * 2, self.config.bert_dim * 3)
        
        # 定义Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.1)
        
        # 定义ReLU激活函数
        self.activation = nn.ReLU()
        
        # 设置设备（使用GPU，如果可用，否则使用CPU）
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def get_encoded_text(self, input_ids, mask):
        """
        获取BERT模型编码后的文本表示
        :param input_ids: 输入的token IDs
        :param mask: attention mask，用于指示哪些token是有效的
        :return: BERT编码后的文本表示（[batch_size, seq_len, bert_dim]）
        """
        bert_encoded_text = self.bert(input_ids=input_ids, attention_mask=mask)[0]
        return bert_encoded_text

    def get_triple_score(self, bert_encoded_text, train):
        """
        计算三元组的得分
        :param bert_encoded_text: BERT编码后的文本表示
        :param train: 是否处于训练模式，控制输出方式
        :return: 三元组得分矩阵
        """
        batch_size, seq_len, bert_dim = bert_encoded_text.size()  # 获取batch size、序列长度和BERT维度

        # [batch_size, seq_len*seq_len, bert_dim]
        # 扩展head_rep的维度并复制到每个尾部位置
        head_rep = bert_encoded_text.unsqueeze(dim=2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len * seq_len, bert_dim)
        
        # 将每个token的表示复制到尾部位置
        tail_rep = bert_encoded_text.repeat(1, seq_len, 1)

        # [batch_size, seq_len*seq_len, bert_dim * 2]
        # 拼接head和tail的表示
        entity_pair = torch.cat([head_rep, tail_rep], dim=-1)

        # [batch_size, seq_len*seq_len, bert_dim * 3]
        # 使用全连接层将实体对映射到一个新的空间
        entity_pair = self.project_matrix(entity_pair)
        
        # 应用Dropout层
        entity_pair = self.dropout_2(entity_pair)
        
        # 使用ReLU激活函数
        entity_pair = self.activation(entity_pair)

        # [batch_size, seq_len*seq_len, num_rel*tag_size]
        # 使用全连接层计算三元组得分，最终输出一个得分矩阵
        matrix_socre = self.relation_linear(entity_pair).reshape(batch_size, seq_len, seq_len, self.config.num_rel, self.config.tag_size)

        if train:
            # 如果是训练模式，返回未转换为标签的得分矩阵
            return matrix_socre.permute(0, 4, 3, 1, 2)
        else:
            # 如果是评估模式，返回最大得分的标签索引
            return matrix_socre.argmax(dim=-1).permute(0, 3, 1, 2)

    def forward(self, data, train=True):
        """
        定义前向传播过程
        :param data: 输入数据（包括input_ids和attention_mask）
        :param train: 是否处于训练模式
        :return: 三元组得分矩阵
        """
        # 获取输入的token IDs和attention mask
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)

        # 获取BERT编码后的文本表示
        bert_encoded_text = self.get_encoded_text(input_ids, attention_mask)
        
        # 应用Dropout层
        bert_encoded_text = self.dropout(bert_encoded_text)

        # 获取三元组得分矩阵
        matrix_score = self.get_triple_score(bert_encoded_text, train)

        return matrix_score  # 返回得分矩阵