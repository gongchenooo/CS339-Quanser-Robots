import numpy as np
import matplotlib.pyplot as plt


a = np.load("Qube/ABC/loss-hyper-example.npy")
b = np.load("Qube/steps/loss_abc_1000.npy")
c = np.load("Qube/steps/loss_maxstep_300.npy")

print(a)
fig = plt.figure(figsize=(8, 5))
plt.title("Loss with Different Max Steps")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot(c, label="300")
plt.plot(a, color='green', label="500")
plt.plot(b, color='red', label="1000")
legend = plt.legend(loc=1)
plt.savefig("figure/loss_steps.png")
plt.show()



