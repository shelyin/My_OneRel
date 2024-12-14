import torch  # 导入 PyTorch 库
from torch.utils.data import Dataset  # 从 PyTorch 中导入 Dataset 类，用于构建数据集
import json  # 导入 json 库，用于处理 JSON 数据
from transformers import BertTokenizer  # 导入 Hugging Face 的 BertTokenizer，用于 BERT 的分词
import numpy as np  # 导入 NumPy 库，用于处理数组

def find_idx(token, target):
    """
    查找目标子串在 token 中的位置
    :param token: 列表类型，表示分词后的句子
    :param target: 要查找的目标子串
    :return: 如果找到，返回目标子串的起始索引，否则返回 -1
    """
    target_length = len(target)
    for k, v in enumerate(token):
        if token[k: k + target_length] == target:
            return k
    return -1  # 如果没有找到目标子串，返回 -1


class REDataset(Dataset):
    def __init__(self, config, file, is_test=False):
        """
        初始化 REDataset 类，用于加载数据集并进行处理
        :param config: 配置文件，包含模型和数据的一些参数
        :param file: 数据文件路径，包含数据集的实际内容
        :param is_test: 是否是测试模式（用于区分训练和测试）
        """
        self.config = config  # 保存配置文件
        with open(file, "r", encoding="utf-8") as f:
            self.data = json.load(f)  # 读取数据集文件并加载为 JSON 格式
        with open(self.config.schema_fn, "r", encoding="utf-8") as fs:
            self.rel2id = json.load(fs)[0]  # 读取关系与 ID 的映射关系
        with open(self.config.tags) as ft:
            self.tag2id = json.load(ft)[1]  # 读取标签与 ID 的映射关系
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)  # 加载 BERT 分词器
        self.is_test = is_test  # 标识是否是测试模式

    def __len__(self):
        """
        返回数据集的大小（样本数量）
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中某个索引的数据
        :param idx: 数据索引
        :return: 处理后的数据，包括句子、三元组、输入 ID、mask 等
        """
        ins_json_data = self.data[idx]  # 获取指定索引的数据
        sentence = ins_json_data["text"]  # 获取句子
        triple = ins_json_data["spo_list"]  # 获取三元组列表（S-P-O）
        token = ['[CLS]'] + list(sentence)[:self.config.max_len] + ['[SEP]']  # 对句子进行分词并添加特殊标记
        token_len = len(token)  # 计算 token 长度

        token2id = self.tokenizer.convert_tokens_to_ids(token)  # 将 token 转换为 token ID
        input_ids = np.array(token2id)  # 转换为 NumPy 数组
        mask = [0] * token_len  # 初始 mask 为全 0
        mask = np.array(mask) + 1  # 将 mask 中的值加 1，使得有效 token 对应 mask 为 1
        mask_len = len(mask)  # 计算 mask 长度
        loss_mask = np.ones((mask_len, mask_len))  # 初始化损失 mask，大小为 (mask_len, mask_len)

        if not self.is_test:
            # 如果不是测试模式，则构建三元组的实体对和关系
            s2po = {}  # 存储实体对和对应的关系
            for spo in triple:  # 遍历三元组
                triple_tuple = (list(spo[0]), spo[1], list(spo[2]))  # 将三元组拆分为 (subject, relation, object)
                sub_head = find_idx(token, triple_tuple[0])  # 找到主语的位置
                obj_head = find_idx(token, triple_tuple[2])  # 找到宾语的位置
                if sub_head != -1 and obj_head != -1:  # 如果都找到了位置
                    sub = (sub_head, sub_head + len(triple_tuple[0]) - 1)  # 获取主语的位置范围
                    obj = (obj_head, obj_head + len(triple_tuple[2]) - 1, self.rel2id[triple_tuple[1]])  # 获取宾语的位置范围及其关系 ID
                    if sub not in s2po:
                        s2po[sub] = []  # 如果主语没有在字典中，初始化其对应的值为空列表
                    s2po[sub].append(obj)  # 将宾语和关系添加到主语的列表中

            if len(s2po) > 0:
                # 如果存在实体对，创建对应的矩阵
                matrix = np.zeros((self.config.num_rel, token_len, token_len))  # 初始化三元组得分矩阵
                for sub in s2po:  # 遍历每个主语
                    sub_head = sub[0]
                    sub_tail = sub[1]
                    for obj in s2po.get((sub_head, sub_tail), []):  # 获取该主语对应的宾语
                        obj_head, obj_tail, rel = obj  # 解析宾语的位置和关系
                        matrix[rel][sub_head][obj_head] = self.tag2id["HB-TB"]  # 设置关系的起始标签
                        matrix[rel][sub_head][obj_tail] = self.tag2id["HB-TE"]  # 设置关系的结束标签
                        matrix[rel][sub_tail][obj_tail] = self.tag2id["HE-TE"]  # 设置宾语尾部标签

                return sentence, triple, input_ids, mask, token_len, matrix, token, loss_mask  # 返回处理后的数据
            else:
                return None  # 如果没有有效的三元组，返回 None
        else:
            # 测试模式下，初始化一个全为零的得分矩阵
            matrix = np.zeros((self.config.num_rel, token_len, token_len))
            return sentence, triple, input_ids, mask, token_len, matrix, token, loss_mask  # 返回测试数据


def collate_fn(batch):
    """
    对一批数据进行处理，返回一个 batch
    :param batch: 输入的批次数据
    :return: 包装好的数据字典
    """
    batch = list(filter(lambda x: x is not None, batch))  # 去除值为 None 的样本
    batch.sort(key=lambda x: x[4], reverse=True)  # 按照 token_len 对样本进行排序，长的在前

    # 解包 batch 中的数据
    sentence, triple, input_ids, mask, token_len, matrix, token, loss_mask = zip(*batch)

    cur_batch = len(batch)  # 当前 batch 的大小
    max_token_len = max(token_len)  # 获取当前 batch 中最长的 token 长度

    # 创建空的张量，用于存放 batch 中的数据
    batch_input_ids = torch.LongTensor(cur_batch, max_token_len).zero_()
    batch_attention_mask = torch.LongTensor(cur_batch, max_token_len).zero_()
    batch_loss_mask = torch.LongTensor(cur_batch, 1, max_token_len, max_token_len).zero_()
    # batch_matrix = torch.LongTensor(cur_batch, 53, max_token_len, max_token_len).zero_()  #CIE数据集 53 是关系数
    # batch_matrix = torch.LongTensor(cur_batch, 11, max_token_len, max_token_len).zero_()  #CTI数据集 11 是关系数
    batch_matrix = torch.LongTensor(cur_batch, 10, max_token_len, max_token_len).zero_()  #Acti数据集 10 是关系数

    # 填充 batch 数据
    for i in range(cur_batch):
        batch_input_ids[i, :token_len[i]].copy_(torch.from_numpy(input_ids[i]))  # 填充 input_ids
        batch_attention_mask[i, :token_len[i]].copy_(torch.from_numpy(mask[i]))  # 填充 attention_mask
        batch_loss_mask[i, 0, :token_len[i], :token_len[i]].copy_(torch.from_numpy(loss_mask[i]))  # 填充 loss_mask
        batch_matrix[i, :, :token_len[i], :token_len[i]].copy_(torch.from_numpy(matrix[i]))  # 填充关系矩阵

    # 返回一个包含多个字段的字典
    return {"sentence": sentence,
            "token": token,
            "triple": triple,
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "matrix": batch_matrix}