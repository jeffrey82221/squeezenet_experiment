from OneCycleTrainFunc import OneCycleTrain
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
pbounds = {
    'squeeze_scale_exp': (1.5, 1.5),  # 2
    'small_filter_rate': (0.5, 0.5),  # 10
    'max_lr_exp': (-4, -2),  # 6
    'max_momentum': (0.8, 0.99),
    'num_epoch': (20, 50)
}
optimizer = BayesianOptimization(
    f=OneCycleTrain,
    pbounds=pbounds,
    random_state=1,
)
try:
    load_logs(optimizer, logs=["./one_cycle_baysian_logs.json"])
except:
    print('no one_cycle_baysian_logs.json')
logger = JSONLogger(path="./one_cycle_baysian_logs.json")

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# run the optimization
optimizer.maximize(
    init_points=300,  # determine according to the boundary of each parameter
    n_iter=8,  # also determine by the boundary of each parameter
)

# access history and result
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("Final Max:", optimizer.max)
