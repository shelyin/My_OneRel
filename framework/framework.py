import json  # 导入json模块，用于读写JSON格式的数据
import numpy as np  # 导入NumPy，用于数组操作
import torch  # 导入PyTorch，用于深度学习模型
from torch.utils.data import DataLoader  # 从PyTorch导入DataLoader，用于批量加载数据
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

from dataloader.dataloader import REDataset, collate_fn  # 导入自定义的数据集类REDataset和collate_fn函数
from models.models import OneRel  # 导入模型类OneRel

class Framework():
    def __init__(self, config):
        """
        初始化Framework类，用于配置模型和训练过程
        :param config: 配置参数
        """
        self.config = config  # 保存配置
        # 读取标签到ID的映射文件
        with open(self.config.tags, "r", encoding="utf-8") as f:
            self.tag2id = json.load(f)[1]  # 加载标签到ID的映射
        # 读取关系ID到名称的映射文件
        with open(self.config.schema_fn, "r", encoding="utf-8") as fs:
            self.id2rel = json.load(fs)[1]  # 加载关系ID到名称的映射
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")  # 定义交叉熵损失函数
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置训练设备为GPU（如果可用），否则使用CPU
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def train(self):
        """
        训练模型的过程
        """
        def cal_loss(predict, target, mask):
            """
            计算损失函数
            :param predict: 模型预测结果
            :param target: 真实标签
            :param mask: 用于计算损失的mask（忽略某些位置）
            :return: 计算后的损失
            """
            loss_ = self.loss_function(predict, target)  # 计算交叉熵损失
            loss = torch.sum(loss_ * mask) / torch.sum(mask)  # 使用mask计算加权损失
            return loss

        # 初始化训练数据集和数据加载器
        train_dataset = REDataset(self.config, self.config.train_file)  # 加载训练集
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_fn)  # 创建训练数据加载器

        # 初始化验证数据集和数据加载器
        dev_dataset = REDataset(self.config, self.config.dev_file)  # 加载验证集
        dev_dataloader = DataLoader(dev_dataset, batch_size=1, collate_fn=collate_fn)  # 创建验证数据加载器

        # 初始化模型并将其移至指定设备（GPU或CPU）
        model = OneRel(self.config).to(self.device)
        # 使用AdamW优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        # 初始化训练过程中的变量
        global_step = 0  # 当前训练步数
        global_loss = 0  # 当前训练损失
        best_epoch = 0  # 最佳模型对应的epoch
        best_f1_score = 0  # 最佳F1分数
        best_recall = 0  # 最佳召回率
        best_precision = 0  # 最佳精确度

        # 开始训练过程
        for epoch in range(self.config.epochs):  # 遍历训练轮次
            print("[{}/{}]".format(epoch+1, self.config.epochs))  # 输出当前训练的轮次
            for data in tqdm(train_dataloader):  # 遍历每一个训练批次
                output = model(data)  # 获取模型的输出（预测）

                optimizer.zero_grad()  # 清空梯度
                loss = cal_loss(output, data["matrix"].to(self.device), data["loss_mask"].to(self.device))  # 计算损失
                global_loss += loss.item()  # 累积损失

                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数
                if (global_step + 1) % 2000 == 0:  # 每2000步打印一次训练信息
                    print("epoch: {} global_step: {} global_loss: {:5.4f}".format(epoch + 1, global_step + 1, global_loss))
                    global_loss = 0  # 重置损失

            if (epoch + 1) % 5 == 0:  # 每5个epoch进行一次验证
                precision, recall, f1_score, predict = self.evaluate(dev_dataloader, model)  # 在验证集上评估模型
                if f1_score > best_f1_score:  # 如果当前模型的F1分数更好
                    best_f1_score = f1_score  # 更新最佳F1分数
                    best_recall = recall  # 更新最佳召回率
                    best_precision = precision  # 更新最佳精确度
                    best_epoch = epoch + 1  # 更新最佳epoch
                    print("save model ......")  # 输出保存模型信息
                    torch.save(model.state_dict(), self.config.checkpoint)  # 保存模型
                    json.dump(predict, open(self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)  # 保存预测结果
                    print("epoch:{} best_epoch:{} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(epoch+1, best_epoch, best_recall, best_precision, best_f1_score))
        print("best_epoch:{} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(best_epoch, best_recall, best_precision, best_f1_score))

    def evaluate(self, dataloader, model):
        """
        在验证集上评估模型的性能
        :param dataloader: 数据加载器
        :param model: 要评估的模型
        :return: 返回精确度、召回率、F1分数和预测结果
        """
        print("eval mode......")  # 输出评估模式信息
        model.eval()  # 设置模型为评估模式（不启用dropout等）
        predict_num, gold_num, correct_num = 0, 0, 0  # 初始化统计信息
        predict = []  # 用于保存预测结果

        def to_ret(data):
            """
            将数据转换为tuple形式
            :param data: 输入数据
            :return: 返回转换后的tuple
            """
            ret = []
            for i in data:
                ret.append(tuple(i))
            return tuple(ret)

        with torch.no_grad():  # 在评估过程中禁用梯度计算
            for data in tqdm(dataloader):  # 遍历验证数据集
                # 获取模型的预测结果
                pred_triple_matrix = model(data, train=False).cpu()[0]
                number_rel, seq_lens, seq_lens = pred_triple_matrix.shape  # 获取预测矩阵的形状
                relations, heads, tails = np.where(pred_triple_matrix > 0)  # 找到预测矩阵中的正值，表示预测的三元组

                token = data["token"][0]  # 获取当前文本的token列表
                gold = data["triple"][0]  # 获取当前文本的真实三元组
                pair_numbers = len(relations)  # 获取预测的三元组个数
                predict_triple = []  # 用于保存预测的三元组
                if pair_numbers > 0:  # 如果有预测的三元组
                    for i in range(pair_numbers):  # 遍历所有预测的三元组
                        r_index = relations[i]
                        h_start_idx = heads[i]
                        t_start_idx = tails[i]
                        if pred_triple_matrix[r_index][h_start_idx][t_start_idx] == self.tag2id["HB-TB"] and i + 1 < pair_numbers:
                            t_end_idx = tails[i + 1]
                            if pred_triple_matrix[r_index][h_start_idx][t_end_idx] == self.tag2id["HB-TE"]:
                                for h_end_index in range(h_start_idx, seq_lens):
                                    if pred_triple_matrix[r_index][h_end_index][t_end_idx] == self.tag2id["HE-TE"]:

                                        subject_head, subject_tail = h_start_idx, h_end_index
                                        object_head, object_tail = t_start_idx, t_end_idx
                                        subject = ''.join(token[subject_head: subject_tail + 1])  # 提取主语
                                        object = ''.join(token[object_head: object_tail + 1])  # 提取宾语
                                        relation = self.id2rel[str(int(r_index))]  # 获取关系
                                        if len(subject) > 0 and len(object) > 0:  # 如果主语和宾语都有效
                                            predict_triple.append((subject, relation, object))  # 保存预测三元组
                                        break
                gold = to_ret(gold)  # 将gold转换为tuple
                predict_triple = to_ret(predict_triple)  # 将预测结果转换为tuple
                gold_num += len(gold)  # 累加gold三元组数量
                predict_num += len(predict_triple)  # 累加预测三元组数量
                correct_num += len(set(gold) & set(predict_triple))  # 计算正确预测的三元组数量
                lack = set(gold) - set(predict_triple)  # 计算缺失的三元组
                new = set(predict_triple) - set(gold)  # 计算新增的三元组
                predict.append({"text": data["sentence"][0], "gold": gold, "predict": predict_triple,
                                "lack": list(lack), "new": list(new)})  # 保存当前文本的预测结果

        # 计算精确度、召回率和F1分数
        precision = correct_num / (predict_num + 1e-10)  # 精确度
        recall = correct_num / (gold_num + 1e-10)  # 召回率
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)  # F1分数
        print("predict_num: {} gold_num: {} correct_num: {}".format(predict_num, gold_num, correct_num))  # 输出统计信息
        model.train()  # 恢复模型为训练模式
        return precision, recall, f1_score, predict  # 返回评估结果