from Model import LeNet, LR, DT, MLP, RF


def determining_original_model(original_name):
    if original_name == "lenet":
        return LeNet()

def determine_attack_model(attack_model_name):
    if attack_model_name == 'LR':
        return LR()
    elif attack_model_name == 'DT':
        return DT()
    elif attack_model_name == 'MLP':
        return MLP()
    elif attack_model_name == 'RF':
        return RF()
    else:
        raise Exception("invalid attack name")