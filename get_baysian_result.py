from OneCycleTrainFunc import OneCycleTrain
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs


def get():
    pbounds = {
        'squeeze_scale_exp': (1.5, 1.5),  # 2
        'small_filter_rate': (0.5, 0.5),  # 10
        'max_lr_exp': (-4, -2),  # 6
        'max_momentum': (0.8, 0.99),
        'num_epoch': (20, 50)
    }
    optimizer = BayesianOptimization(
        f=OneCycleTrain,
        pbounds=pbounds
    )
    load_logs(optimizer, logs=["./one_cycle_baysian_logs.json"])

    res_list = []
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        res_list.append(res)

    # print("Final Max:", optimizer.max)
    return res_list
