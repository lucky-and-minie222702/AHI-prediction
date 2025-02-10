from data_functions import *
import sys
from os import path

num_p = 33
info = open(path.join("data", "info.txt"), "r").readlines()
p_list = []
no_spo2 = []

for s in info:
    s = s[:-1:]
    if "*" in s:
        p_list.append(int(s[1::]))
        no_spo2.append(int(s[1::]))
    else:
        p_list.append(int(s))

