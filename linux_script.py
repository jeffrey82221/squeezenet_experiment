import os
lr = 0.0001
# for scale in [1.5, 1.0]:
for scale, small_filter_rate in [(30.0, 0.5), (50.0, 0.5)]:
    for count in range(0, 30):
        command = "python SqueezeNetExperiment.py %s %s %s" % (scale, small_filter_rate, lr)
        os.system(command + " >> squeeze_log/%s_%s_%s_%s.log" % (scale, small_filter_rate, lr, count))
