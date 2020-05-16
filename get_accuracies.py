
'''
This code print the accuracy of models with a given parameter setup
'''
import sys


def get(scale, rate, count):
    f = open("squeeze_log/" + str(scale) + "_" + str(rate) + "_0.0001_" + str(count) + ".log")
    last_line = list(f)[-1]
    acc = float(last_line.split(" ")[-2])
    return acc


scale = float(sys.argv[1])
small_filter_rate = float(sys.argv[2])
try:
    count = int(sys.argv[3])
except:
    count = 30

for i in [get(scale, small_filter_rate, c) for c in range(count)]:
    print(i)
