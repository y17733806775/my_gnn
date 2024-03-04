from models.spa_tune import SPATune
from models.spa_spos_search import SPASPOSSearch
from utils import get_metrics, get_logger, get_name, count_parameters_in_MB
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch
import time
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, rand, space_eval
from os import mkdir, makedirs
from os.path import exists
from models.genotypes_new import NA_PRIMITIVES, LC_PRIMITIVES, LF_PRIMITIVES, SEQ_PRIMITIVES
# DEBUG
from hyperopt.pyll.stochastic import sample
from pprint import pprint
from itertools import product
from sortedcontainers import SortedDict
from tqdm import tqdm

EPOCH_TEST = {"icews14/": 1,
              "icews05-15/": 10,
              "gdelt/": 30,
              "wikidata11k/": 50}


class Trainer(object):
    cnt_tune = 0

    def __init__(self, args, dataset_info_dict, train_loader, evaluate_loader, device):
        self.args = args
        self.device = device
        self.dataset_info_dict = dataset_info_dict
        self.train_loader = train_loader
        self.evaluate_loader = evaluate_loader
        self.optimizer = None
        self.scheduler = None
        self.search_space = None
        self.logger = None

    def train(self):
        name = get_name(self.args)
        print(name)
        log_dir = f'{self.args.log_dir}{self.args.dataset}{self.args.train_mode}/'
        if not exists(log_dir):
            mkdir(log_dir)
        self.logger = get_logger(name, log_dir)
        self.logger.info(self.args)
        writer = SummaryWriter(self.args.tensorboard_dir + self.args.dataset + name)
        model = SPATune(self.args, self.dataset_info_dict, self.device)
        model = model.cuda()
        self.logger.info("Parameter size = %fMB", count_parameters_in_MB(model))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=0.0001)

        best_val_mrr, best_test_mrr = 0.0, 0.0
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            training_loss = self.train_epoch(epoch, model, architect=None, lr=None, mode="train")
            valid_mrr = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_val_mrr:
                early_stop_cnt = 0
                best_val_mrr = valid_mrr
                test_mrr = self.evaluate_epoch(epoch, model, split="test")
                if test_mrr > best_test_mrr:
                    best_test_mrr = test_mrr
                    self.logger.info("Success")
                    # torch.save(model.state_dict(), f'{args.saved_model_dir}{name}.pth')
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 10:
                self.logger.info("Early stop!")
                self.logger.info(best_test_mrr)
                break
            writer.add_scalar('Loss/train', training_loss, epoch)
            writer.add_scalar('MRR/test', best_test_mrr, epoch)
    # 没用
    def evaluate_epoch(self, current_epoch, model, split="valid", evaluate_ws=False, mode=None):
        rank_list = []
        loss_list = []
        model.eval()
        with torch.no_grad():
            for batch_idx, timestamps in enumerate(self.evaluate_loader):
                if mode == "spos_train":
                    model.ent_encoder.ops = model.ent_encoder.generate_single_path()
                rank, loss = model.evaluate(timestamps, split, evaluate_ws=evaluate_ws)
                # 模型评估
                rank_list.append(rank)
                if split == 'valid' or split == 'train':
                    loss_list.append(loss.item())
                else:
                    loss_list.append(loss)
            if split == "train":
                self.logger.info(
                    '[Epoch:{} | {}]: Loss:{:.4}'.format(
                        current_epoch, split.capitalize() + ('_WS' if evaluate_ws else ""), np.mean(loss_list)))
                return np.mean(loss_list)
            else:
                all_ranks = torch.cat(rank_list)
                mrr, hit_1, hit_3, hit_10 = get_metrics(all_ranks)
                metrics_dict = {'mrr': mrr, 'hit_10': hit_10, 'hit_3': hit_3, 'hit_1': hit_1}
                metrics_result = {k: v.item() for k, v in metrics_dict.items()}
                # self.logger.info(
                #     '[Epoch:{} | {}]: {} Loss:{:.4}'.format(current_epoch, split.capitalize(), split.capitalize(), np.mean(loss_list)))
                # self.logger.info('[Epoch:{} | {}]: Loss:{:.4}, MRR:{:.3}, Hits@10:{:.3}, Hits@3:{:.3}, Hits@1:{:.3}'.format(
                #     current_epoch, split.capitalize() + ('_WS' if evaluate_ws else ""), np.mean(loss_list),
                #     metrics_result['mrr'], metrics_result['hit_10'],
                #     metrics_result['hit_3'],
                #     metrics_result['hit_1']))
                print("epoch:"+" "+str(current_epoch)+" | "+str(split.capitalize()))
                print("loss: "+str(np.mean(loss_list))+" MRR: "+str(metrics_result['mrr']))
                print(str(metrics_result['hit_1'])+" "+str(metrics_result['hit_3'])+" "+str(metrics_result['hit_10']))
                return metrics_result['mrr'], metrics_result, np.mean(loss_list)

    def train_epoch(self, current_epoch, model, architect=None, lr=None, mode='NONE'):
        train_loss_list = []
        for batch_idx, train_timestamps in enumerate(self.train_loader):
            if mode == "spos_search":
                train_loss = model(train_timestamps)
                train_loss_list.append(train_loss.item())
            else:
                model.train()
                self.optimizer.zero_grad()
                train_loss = model(train_timestamps)
                train_loss_list.append(train_loss.item())
                train_loss.backward()
                self.optimizer.step()
                # print("epoch:"+""+str(current_epoch) +" "+str(self.args.train_mode.capitalize()))
                # print("Train Loss"+str(np.mean(train_loss_list)))
        #self.logger.info('[Epoch:{} | {}]: Train Loss:{:.4}'.format(current_epoch, self.args.train_mode.capitalize(),
                                                                    #np.mean(train_loss_list)))
        return np.mean(train_loss_list)

    def train_parameter(self):
        Trainer.cnt_tune += 1
        icews14_result = "rgcn||sa||lc_concat||rgat_vanilla||identity||lc_concat||compgcn_rotate||identity||lf_mean"
        model = SPATune(self.args, self.dataset_info_dict, self.device, icews14_result)
        # 这里定义了model
        model = model.cuda()
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)
        best_valid_mrr, best_test_mrr = 0.0, 0.0
        best_metrics_result = {'mrr': 0.0, 'hit_10': 0.0, 'hit_3': 0.0, 'hit_1': 0.0}
        early_stop_cnt = 0
        for epoch in range(1, self.args.max_epoch + 1):
            loss = self.train_epoch(epoch, model, mode="tune")
            valid_mrr, metrics_dict_valid, _ = self.evaluate_epoch(epoch, model, split="valid")
            if valid_mrr > best_valid_mrr:
                early_stop_cnt = 0
                best_valid_mrr = valid_mrr
                if self.args.train_mode == "tune" and epoch > EPOCH_TEST[self.args.dataset]:
                    test_mrr, metrics_dict_test, _ = self.evaluate_epoch(epoch, model, split="test")
                    if test_mrr > best_test_mrr:
                        best_test_mrr = test_mrr
                        best_metrics_result = metrics_dict_test
                        #self.logger.info("Success")
                        print("success")
            else:
                early_stop_cnt += 1
            if early_stop_cnt > 25 or epoch == self.args.max_epoch:
                # self.logger.info("Early stop!")
                # self.logger.info(f'{best_valid_mrr} {self.args.genotype}')
                print("earlystop")
                break
            self.scheduler.step(best_valid_mrr)
        print("The best MRR: " + str(best_metrics_result['mrr']))
        print("The best hit: " + str(best_metrics_result['hit_1']) + " " + str(best_metrics_result['hit_3']) + " " + str(best_metrics_result['hit_10']))
        return {'loss': -best_valid_mrr, 'status': STATUS_OK} if self.args.train_mode == "search" else {'loss': -best_valid_mrr, 'test_mrr':best_test_mrr,'status': STATUS_OK}
