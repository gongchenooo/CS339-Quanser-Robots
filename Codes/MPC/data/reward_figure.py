import numpy as np
import matplotlib.pyplot as plt


r1 = np.load("CartPoleSwing/ABC/reward-abc.npy")
r2 = np.load("CartPoleSwing/Random/reward-random.npy")
r3 = np.load("CartPoleSwing/Cannon/reward-cannnon.npy")

a = []
b = []
c = []

for i in range(50):
    a.append(np.average(r1[4*i:4*(i+1)]))
    b.append(np.average(r2[4*i:4*(i + 1)]))
    c.append(np.average(r3[4 * i:4 * (i + 1)]))


fig = plt.figure(figsize=(8, 5))
plt.title("Traning Reward with Different Optimization Methods")
plt.xlabel("Process")
plt.ylabel("Training Reward")

plt.plot(b, color='green', label="Random")
plt.plot(a, label="ABC")
plt.plot(c, color='red', label="Cannon")
legend = plt.legend(loc=1)
plt.savefig("figure/reward_cartpoleswing.png")
plt.show()


