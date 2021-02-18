import numpy as np
import matplotlib.pyplot as plt

r1 = np.load("Qube/batch/ABC_batchSize=32_rewards_lst.npy")
r2 = np.load("Qube/batch/reward-batch-128.npy")
print(r2)
r3 = np.load("Qube/hyper-example/reward-hyper-example.npy")

a = []
b = []
c = []

for i in range(50):
    a.append(np.average(r1[4*i:4*(i+1)]))
    b.append(np.sum(r2[4 * i:4 * (i + 1)])/4)
    c.append(np.average(r3[4 * i:4 * (i + 1)]))


fig = plt.figure(figsize=(8, 5))
plt.title("Traning Reward with Different Batch Size")
plt.xlabel("Process")
plt.ylabel("Training Reward")

plt.plot(c, color='green', label="512")
plt.plot(a, label="32")
plt.plot(b, color='red', label="128")
legend = plt.legend(loc=4)
plt.savefig("Qube/figure/batch_size.png")


