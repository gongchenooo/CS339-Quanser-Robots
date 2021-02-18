import numpy as np
import re

a = []
pattern = "Reward:.*,"
with open("BallBalancer/ABC/records.TXT") as file:
    line = file.readline()
    while line:
        if "Reward" in line:
            b = re.search(pattern, line).span()
            value = float(line[b[0]+7:b[1]-1])
            a.append(value)
        line = file.readline()

a = np.array(a)
print(a)
np.save("BallBalancer/ABC/reward_abc.npy", a)