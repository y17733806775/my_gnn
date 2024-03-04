from preprocess import load_dataset, TemporalDataset
from collections import OrderedDict
from json import dump, load
from torch.utils.data import DataLoader
from utils import initialize_seed
from trainer import Trainer
from time import strftime
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Search to Pass Messages for Temporal Knowledge Completion")
    parser.add_argument("--dataset", type=str, default="icews14/")
    parser.add_argument("--train_mode", type=str, default="tune",
                        choices=["search", "tune", "train", "debug"])
    parser.add_argument("--search_mode", type=str, default="",
                        choices=["random", "spos", "spos_search"])
    parser.add_argument("--encoder", type=str, default="")
    parser.add_argument("--score_function", type=str, default="complex")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=22)
    parser.add_argument("--gnn_layer_num", type=int, default=3)
    parser.add_argument("--rnn_layer_num", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    # base vector config
    parser.add_argument("--base_num", type=int, default=0)
    # RGAT config
    parser.add_argument("--head_num", type=int, default=4)
    # CompGCN config
    parser.add_argument("--comp_op", type=str, default="corr")
    parser.add_argument("--sampled_dataset", type=bool, default=False)
    # Optimizer config
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0009386912609690407)
    # Dynamic config
    parser.add_argument("--train_seq_len", type=int, default=8)
    parser.add_argument("--test_seq_len", type=int, default=8)
    parser.add_argument("--rec_only_last_layer", type=bool, default=True)
    parser.add_argument("--use_time_embedding", type=bool, default=False)
    parser.add_argument("--seq_head_num", type=int, default=2)
    # search config
    parser.add_argument("--baseline_sample_num", type=int, default=30)
    parser.add_argument("--search_run_num", type=int, default=1)
    parser.add_argument("--search_max_epoch", type=int, default=800)
    parser.add_argument("--min_learning_rate", type=float, default=0.001)
    parser.add_argument("--unrolled", action='store_true', default=False)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument("--arch_learning_rate", type=float, default=0.01)
    parser.add_argument("--min_arch_learning_rate", type=float, default=0.0005)
    parser.add_argument("--arch_weight_decay", type=float, default=1e-3)
    # spos config
    parser.add_argument("--arch_sample_num", type=int, default=1000)
    parser.add_argument("--stand_alone_path", type=str, default='')
    # fine-tune config
    parser.add_argument("--tune_sample_num", type=int, default=20)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--negative_sampling_num", type=int, default=500)
    parser.add_argument("--isolated_change", type=bool, default=False)
    parser.add_argument("--positive_fact_num", type=int, default=3000)
    parser.add_argument("--dataset_dir", type=str, default="datasets/")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--time_log_dir", type=str, default="")
    parser.add_argument("--tensorboard_dir", type=str, default="tensorboard/")
    parser.add_argument("--saved_model_dir", type=str, default="saved_models/")
    parser.add_argument("--weight_path", type=str, default='')
    parser.add_argument("--fixed_ops", type=str, default='')
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--search_res_dir", type=str, default="searched_res/")
    parser.add_argument("--tune_res_dir", type=str, default="tune_res/")
    parser.add_argument("--search_res_file", type=str, default="")
    parser.add_argument("--arch", type=str, default="")
    parser.add_argument("--inv_temperature", type=float, default=0.1)
    args = parser.parse_args()
    dataset_info_dict = load_dataset(args.dataset_dir + args.dataset)
    train_dataset = TemporalDataset(dataset_info_dict['train_timestamps'], toy=args.sampled_dataset)
    valid_dataset = TemporalDataset(dataset_info_dict['valid_timestamps'], toy=args.sampled_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    evaluate_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size * 2,
                                 shuffle=True, num_workers=0)

    initialize_seed(args.random_seed)
    device = torch.device('cuda:0')
    trainer = Trainer(args, dataset_info_dict, train_loader, evaluate_loader, device)
    # trainer实例化

    icews14_result =["rgcn||sa||lc_concat||rgat_vanilla||identity||lc_concat||compgcn_rotate||identity||lf_mean"]
    trainer.train_parameter()

if __name__ == '__main__':
    main()