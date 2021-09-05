# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, train_with_pgd, train_with_free, train_with_FGSM
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--adversarial_model', default='Baseline', type=str, help='choose a adversarial strategy = Baseline, PGD, Free, FGSM')
    parser.add_argument('--pgd_steps', default=7, type=int, help='PGD steps')
    parser.add_argument('--minibatch_replays', default=10, type=int, help='minibatch replays')
    return parser.parse_args()

def main():
    args = get_args()
    dataset = 'THUCNews'  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    model_name = 'TextCNN'

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters())
    if args.adversarial_model == 'Baseline':
        print('adversarial strategy: none')
        train(config, model, train_iter, dev_iter, test_iter)
    elif args.adversarial_model == 'PGD':
        print('adversarial strategy: PGD')
        pgd_steps = args.pgd_steps
        train_with_pgd(config, model, train_iter, dev_iter, test_iter, pgd_steps)
    elif args.adversarial_model == 'Free':
        print('adversarial strategy: Free')
        minibatch_replays = args.minibatch_replays
        train_with_free(config, model, train_iter, dev_iter, test_iter, minibatch_replays)
    elif args.adversarial_model == 'FGSM':
        print('adversarial strategy: FGSM with random initial')
        train_with_FGSM(config, model, train_iter, dev_iter, test_iter)

if __name__ == '__main__':
    main()