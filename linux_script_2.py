import os
lr = 0.0001
#for scale in [1.5, 1.0]:
for scale, small_filter_rate in [(6.0, 0.3), (6.0, 0.7), (10.0, 0.5), (10.0, 0.3),
	(10.0, 0.7), (14.0, 0.5), (14.0, 0.7), (14.0, 0.3), (20.0, 0.5), (25.0, 0.5), (3.00, 0.5)]:
    for count in range(0,30):
        command = "python SqueezeNetExperiment.py %s %s %s" % (scale,small_filter_rate,lr)
        os.system(command + " >> squeeze_log/%s_%s_%s_%s.log" % (scale,small_filter_rate,lr, count))
