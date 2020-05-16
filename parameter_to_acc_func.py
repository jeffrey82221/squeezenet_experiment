import os


def get():
    f = open("current_log")
    last_line = list(f)[-1]
    acc = float(last_line.split(" ")[-2])
    return acc


def parameter_to_accuracy(squeeze_scale_exp, small_filter_rate, lr_exp):
    squeeze_scale = 10**squeeze_scale_exp
    lr = 10**lr_exp
    command = "python SqueezeNetExperiment.py %s %s %s" % (squeeze_scale, small_filter_rate, lr)
    os.system(command + " >> current_log")
    acc = get()
    return acc
