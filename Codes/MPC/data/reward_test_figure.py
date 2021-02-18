import numpy as np
import matplotlib.pyplot as plt


r1 = np.load("Qube/hyper-example/reward-hyper-example-test.npy")
r2 = np.load("Qube/steps/reward_maxstep_1000_test.npy")
r3 = np.load("Qube/steps/reward_maxstep_300_test.npy")
a = []
b = []
c = []

for i in range(10):
    a.append(np.average(r1[4*i:4*(i+1)]))
    b.append(np.average(r2[4 * i:4 * (i + 1)]))
    c.append(np.average(r3[4* i:4 * (i + 1)]))

fig = plt.figure(figsize=(8, 5))
plt.title("Test Reward with Different Max Steps")
plt.xlabel("Process")
plt.ylabel("Test Reward")

plt.plot(c, label="300")
plt.plot(a, color='green', label="500")
plt.plot(b, color='red', label="1000")
legend = plt.legend(loc=4)
plt.savefig("figure/reward_steps_test.png")
plt.show()


