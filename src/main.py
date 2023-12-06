# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import RecWithContrastiveLearningDataset
from models import OfflineItemSimilarity, SASRecModel
from trainers import KECLRecTrainer
from utils import check_path, set_seed, get_user_seqs, EarlyStopping, get_threshold_value


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='ml-1m', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument('--tail_test', action='store_true')
    parser.add_argument('--random_augment_combination', default='SM', type=str,
                        help='SM SR IR... choose any two operators from them to combine.')
    parser.add_argument('--similarity_model_name', default='HNSW', type=str,
                        help="Method to generate item similarity score. choices: \
                        cosine_similarity, HNSW, Random")
    parser.add_argument('--base_augment_type', default='random', type=str,
                        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).")
    parser.add_argument("--tao", type=float, default=0.1, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.1, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.1, help="reorder ratio for reorder operator")
    parser.add_argument("--substitute_rate", type=float, default=0.1,
                        help="substitute ratio for substitute operator")
    parser.add_argument("--insert_rate", type=float, default=0.1,
                        help="insert ratio for insert operator")
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1,
                        help="maximum insert items per position for insert operator - not studied")

    # contrastive learning task args
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--n_views', default=2, type=int, metavar='N',
                        help='Number of augmented data for each sequence - not studied.')

    # model args
    parser.add_argument("--model_name", default='KECLSR', type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.1, \
                        help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, \
                        help="weight of contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)  # 检查output路径是否存在,不存在则创建

    #
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)
    # print(user_seq)
    """
    user_seq是交互序列，是一个包含物品ID的整数列表。
        例如，如果用户1的交互序列是 [2, 5, 6]，用户2的交互序列是 [1, 3, 7, 2]，
        那么user_seq就是一个包含这两个列表的列表：[[2, 5, 6], [1, 3, 7, 2]]。
    valid_rating_matrix是不含有最后两个交互物品的01交互矩阵
    test_rating_matrix是不含有最后一个交互物品的01交互矩阵
    """

    # ??
    args.item_size = max_item + 1
    # max_item为3706
    print(f'max_item = {max_item}')  # 3707
    # sys.exit(0)

    args.mask_id = max_item + 1

    args.threshold = get_threshold_value(args.data_file)
    print(f'threshold = {args.threshold}')

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    show_args_info(args)

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_model_path = os.path.join(args.data_dir, \
                                              args.data_name + '_' + args.similarity_model_name + '_similarity.pkl')
    # ../data/ml-1m_ItemCF_similarity.pkl

    # model_name需要传递对应的相似度查询信息
    offline_similarity_model = OfflineItemSimilarity(args, data_file=args.data_file,
                                                     similarity_path=args.similarity_model_path,
                                                     model_name=args.similarity_model_name,
                                                     dataset_name=args.data_name)
    args.offline_similarity_model = offline_similarity_model

    # training data for node classification
    train_dataset = RecWithContrastiveLearningDataset(args,
                                                      user_seq[:int(len(user_seq))],
                                                      data_type='train')

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = SASRecModel(args=args)

    trainer = KECLRecTrainer(model, train_dataloader, eval_dataloader,
                             test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f'Train KECLSR')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


if __name__ == '__main__':
    main()
