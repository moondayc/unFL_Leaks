import pickle

from Model import LeNet, LR, DT, MLP, RF, LR_2, LR_2_without_scaler, LR_without_scaler, SimpleCNN


def determining_original_model(original_name, dataset_name):
    if original_name == "lenet":
        return LeNet()
    elif original_name == 'LR':
        return LR(dataset_name)
    elif original_name == 'LR_without':
        return LR_without_scaler(dataset_name)
    elif original_name == 'simpleCNN':
        return SimpleCNN()
    else:
        raise Exception("invalid original model")

def determine_attack_model(attack_model_name):
    if attack_model_name == 'LR':
        return LR_2()
    elif attack_model_name == 'LR_without':
        return LR_2_without_scaler()
    elif attack_model_name == 'DT':
        return DT()
    elif attack_model_name == 'MLP':
        return MLP()
    elif attack_model_name == 'RF':
        return RF()
    else:
        raise Exception("invalid attack name")
