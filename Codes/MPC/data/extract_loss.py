import numpy as np
import re

data = []
a = []
pattern = "Training Loss:.*,"
with open("Qube/Cannon/records.TXT") as file:
    line = file.readline()
    while line:
        if "Training Loss" in line:
            b = re.search(pattern, line).span()
            value = float(line[b[0]+15:b[1]-1])
            a.append(value)
        line = file.readline()

a = a[1:]
a = np.array(a)
print(a)
np.save("Qube/Cannon/loss_cannon.npy", a)
