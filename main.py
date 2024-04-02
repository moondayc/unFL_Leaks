import time

from sklearn.linear_model._cd_fast import ConvergenceWarning

from argument import get_args
from exp import ModelTrainer, AttackModelTrainer
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main(args):
    ModelTrainer(args)
    AttackModelTrainer(args)


if __name__ == "__main__":
    time1 = time.time()
    args = get_args()
    main(args)
    time2 = time.time()
    print("实验用时：{:.2f}".format(time2 - time1))
