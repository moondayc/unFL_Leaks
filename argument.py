import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--dataset_name", default="nmist",
                        choices=["nmist"],
                        help="the training dataset")
    # 初始模型结构和攻击模型结构
    parser.add_argument("-om", "--original_model_name", default="lenet",
                        choice=["DT", "lenet"])  # 原始模型的类别
    parser.add_argument("-am", "--attack_model_name", default="DT",
                        choices=['DT', 'MLP', 'LR', 'RF'], help="attack model")  # 攻击模型的类别
    # ------- 本地模型训练参数
    parser.add_argument("-batch", "--local_batch_size", default=20,
                        help="number of batches in machine learning.")
    parser.add_argument('-epoch', '--local_epoch', default=20,
                        help="epochs of client-side local training")

    # ------- 联邦学习相关参数
    parser.add_argument("-cn", "--client_number", default=10,
                        help="number of clients participating in one round of aggregation")
    parser.add_argument("-acn", "--all_client_number", default=20,
                        help="number of all clients participating in aggregation")
    parser.add_argument('-max', "--max_aggregation_round", default=100,
                        help="the maximum aggregation count")
    parser.add_argument('-d', "--decimal_places", default=8)

    # ----- 训练的相关参数
    parser.add_argument('--round_number', type=int, default=10,
                        help="Number of round of original model training")
    # parser.add_argument('--shadow_set_size', type=int, default=2000,
    #                     help="Number of shadow model training samples")

    # ------- unlearning 的相关参数
    # parser.add_argument("-un", "--unlearning round", default=10,
    #                     help="the number of rounds of unlearning training")  # 默认每轮unlearning去掉一个
    args = vars(parser.parse_args())
    return args
