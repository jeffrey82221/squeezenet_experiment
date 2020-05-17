# https://github.com/fmfn/BayesianOptimization
# TODO: steps :
# [V] 1. construct the black box function:
#    - using the squeezenet_experiment.py
#    - run using os.script
#    - obtain result from a common .log file
# [V] 2. test the constructed black box function
# [V] 3. determine the parameters and boundaries for each parameter
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from SqueezeNetFunction import SqueezeNetFunction
# Bounded region of parameter space
pbounds = {
    'squeeze_scale_exp': (-1., 1.5),  # 2
    'small_filter_rate': (0., 1.),  # 10
    'lr_exp': (-5., -3.)            # 6
}

optimizer = BayesianOptimization(
    f=SqueezeNetFunction,
    pbounds=pbounds,
    random_state=7,
)
try:
    load_logs(optimizer, logs=["./baysian_logs.json"])
except:
    print('no baysian_logs')

# subsribing the optimizing history
logger = JSONLogger(path="./baysian_logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# run the optimization
optimizer.maximize(
    init_points=300,  # determine according to the boundary of each parameter
    n_iter=8,      # also determine by the boundary of each parameter
)
# access history and result

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("Final Max:", optimizer.max)
