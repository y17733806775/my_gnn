from os.path import join
from numpy import asarray, unique, reshape, concatenate
from collections import defaultdict
from dataclasses import dataclass, field
from collections import namedtuple
from torch.utils.data import Dataset
from utils import compute_norm
import dgl
import torch
from random import sample
from pprint import pprint


@dataclass
class TimestampFact:
    train: list = field(default_factory=list)
    valid: list = field(default_factory=list)
    test: list = field(default_factory=list)
    # 三个一维列表

@dataclass
class EvalFact:
    head: dict = field(default_factory=lambda: defaultdict(list))
    tail: dict = field(default_factory=lambda: defaultdict(list))
    # 两个字典 值是列表
    # {
    #     'key1':[]
    #     'key2':[]
    # }
    # {
    #     'key1': []
    #     'key2': []
    # }

@dataclass
class TimestampGraph:
    train: dgl.DGLGraph()
    valid: dgl.DGLGraph()
    test: dgl.DGLGraph()


MODE = ['train', 'valid', 'test']

def load_dataset(dataset_path):
    type_id2name_dict = load_type_id2name(dataset_path)
    entity_num = len(type_id2name_dict["entity"])
    relation_num = len(type_id2name_dict["relation"])
    train_timestamp_set = set()
    valid_timestamp_set = set()

    time2fact_dict = defaultdict(TimestampFact)
    # defaultdict(TimestampFact)创建一个实例化对象，当访问的键不存在时，默认返回TimestampFact对象
    train_sr2o_dict = defaultdict(EvalFact)
    sr2o_dict = defaultdict(EvalFact)
    for filename, mode in zip(['train.txt', 'valid.txt', 'test.txt'], ['train', 'valid', 'test']):
        with open(join(dataset_path, filename), 'r') as fr:
            for line in fr:
                head, rel, tail, time = tuple(map(int, line.split()))
                # map()将int()函数应用到分割后的列表元素上，返回一个迭代器，tuple()再转换回来
                time2fact_dict[time].__getattribute__(mode).append((head, rel, tail))
                # __getattribute__(mode)把mode代表的字符串当作参数，这里mode是字典名字
                # {
                #     0:
                #         {
                #             train:[(12, 4, 234)]
                #             valid:[]
                #             test: []
                #         }
                #     1:
                #         {
                #             train: []
                #             valid: []
                #             test:  []
                #         }
                # }
                if mode == 'train':
                    train_sr2o_dict[time].tail[(head, rel)].append(tail)
                    train_sr2o_dict[time].head[(rel, tail)].append(head)
                    # {
                    #     0:
                    #         {
                    #             (rel1, tail2)：head1
                    #         }
                    #         {
                    #             (head1, rel1): tail1
                    #         }
                    #     1:  {}{}
                    # }
                    train_timestamp_set.add(time)
                elif mode =='valid':
                    valid_timestamp_set.add(time)
                sr2o_dict[time].tail[(head, rel)].append(tail)
                sr2o_dict[time].head[(rel, tail)].append(head)
    dataset_info_dict = {
        'entity_num': entity_num,
        'relation_num': relation_num,
        'train_timestamps': list(train_timestamp_set),
        'valid_timestamps': list(valid_timestamp_set),
        'type_id2name_dict': type_id2name_dict,
        'time2fact_dict': time2fact_dict,
        'time2graph_dict': get_timestamp_graph(time2fact_dict),
        'negative_sr2o_dict': train_sr2o_dict,
        'filter_sr2o_dict': sr2o_dict
    }
    return dataset_info_dict


def get_timestamp_graph(time2fact_dict):
    time2graph_dict = defaultdict(TimestampGraph)
    for timestamp in time2fact_dict:
        graph_list = allfact2graph(time2fact_dict[timestamp])
        time2graph = TimestampGraph(graph_list[0], graph_list[1], graph_list[2])
        time2graph_dict[timestamp] = time2graph
    return time2graph_dict
    # {
    #     0:
    #         {
    #            g
    #         }
    #     1:
    #         {
    #
    #         }
    # }


def allfact2graph(fact):
    all_fact_list = []
    all_graph_list = []
    for mode in MODE:
        if fact.__getattribute__(mode):
            all_fact_list.append(asarray(fact.__getattribute__(mode)))
            # [[(10,1,2),...],[(1,2,3),...],[(3,2,4),...]]
    all_fact_array = concatenate(all_fact_list, axis=0)
    # [[1,2,3],[4,5,6],...]
    raw_all_graph_id, new_graph_id = unique((all_fact_array[:, 0], all_fact_array[:, 2]), return_inverse=True)
    # 第一个数组按顺序返回列表中唯一出现的元素 第二个数组返回列表中所有元素对 唯一元素列表 的映射
    new_src_id, new_dst_id = reshape(new_graph_id, (2, -1))
    for mode in MODE:
        g = dgl.DGLGraph()
        g.add_nodes(len(raw_all_graph_id))
        train_fact_num, valid_fact_num = len(fact.train), len(fact.valid)
        if mode == 'train':
            g.add_edges(new_src_id[:train_fact_num], new_dst_id[:train_fact_num])
            g.edata['id'] = torch.LongTensor(all_fact_array[:, 1][:train_fact_num])
        elif mode == 'valid':
            g.add_edges(new_src_id[train_fact_num:train_fact_num + valid_fact_num],
                        new_dst_id[train_fact_num:train_fact_num + valid_fact_num])
            g.edata['id'] = torch.LongTensor(all_fact_array[:, 1][train_fact_num:train_fact_num + valid_fact_num])
        else:
            g.add_edges(new_src_id[train_fact_num + valid_fact_num:],
                        new_dst_id[train_fact_num + valid_fact_num:])
            g.edata['id'] = torch.LongTensor(all_fact_array[:, 1][train_fact_num + valid_fact_num:])
        norm = compute_norm(g)
        g.ndata['id'] = torch.LongTensor(raw_all_graph_id).view(-1, 1)
        g.ndata['norm'] = norm.unsqueeze(1)
        g.ids = {}
        g.rids = {}
        in_graph_idx = 0
        in_graph_rel_idx = 0
        # graph.ids: node id in the entire node set -> node index
        for idx in raw_all_graph_id:
            g.ids[in_graph_idx] = idx
            in_graph_idx += 1
        all_graph_list.append(g)
    return all_graph_list


class TemporalDataset(Dataset):
    def __init__(self, time_list, toy=False):
        time_list.sort()
        if toy:
            time_list = time_list[:len(time_list)//2]
        self.times = asarray(time_list)

    def __getitem__(self, index):
        return self.times[index]

    def __len__(self):
        return len(self.times)


def load_type_id2name(dataset_path):
    type_id2name_dict = defaultdict(dict)
    for filename, type in zip(['entity2id.txt', 'relation2id.txt'], ['entity', 'relation']):
        with open(join(dataset_path, filename), 'r', encoding='gb18030', errors = 'ignore') as fr:
            for line in fr:
                name, idx = tuple(map(str, line.split('\t')))
                type_id2name_dict[type][int(idx)] = name
    return type_id2name_dict
    # 二层字典 外层是类型 内层是id到name
    # {
    #     'entity': {
    #         0: 'entity_name_0',
    #         1: 'entity_name_1',
    #         ...
    #     },
    #     'relation': {
    #         0: 'relation_name_0',
    #         1: 'relation_name_1',
    #         ...
    #     }
    # }
