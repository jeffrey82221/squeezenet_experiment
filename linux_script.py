import os
lr = 0.0001
#for scale in [1.5, 1.0]:
for scale, small_filter_rate in [(1.5,0.5), (0.8,0.7), (0.9, 0.7), (1.0, 0.7),
	(1.1, 0.7), (1.2, 0.7), (1.5, 0.7), (4.0, 0.7), (5.0, 0.7), (6.0, 0.7)]:
    for count in range(0,30):
        command = "python SqueezeNetExperiment.py %s %s %s" % (scale,small_filter_rate,lr)
        os.system(command + " >> squeeze_log/%s_%s_%s_%s.log" % (scale,small_filter_rate,lr, count))
