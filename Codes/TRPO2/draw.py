import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

'''
r1 = np.load("Qube/ABC/reward-hyper-example.npy")
r2 = np.load("Qube/Cannon/reward_cannon.npy")
r3 = np.load("Qube/Random/rewardslist_ran.npy")

a = []
b = []
c = []


for i in range(50):
    a.append(np.average(r1[2*i:2*(i+1)]))
    b.append(np.average(r2[4 * i:4 * (i + 1)]))
    c.append(np.average(r3[4 * i:4 * (i + 1)]))


fig = plt.figure(figsize=(8, 5))
plt.title("Traning Reward with Different Optimization Method")
plt.xlabel("Process")
plt.ylabel("Training Reward")

plt.plot(a, label="ABC")
plt.plot(c, color='green', label="Random")
plt.plot(b, color='red', label="Cannon")
legend = plt.legend(loc=1)
plt.savefig("figure/reward_qube.png")
plt.show()
'''
def rewards():
    a = pd.read_csv('logs/Qube-100-v0/lr=0.00030_batchsize=32_gamma=0.990_tau=0.950.csv')
    a = a['rewards'].values[0:500]

    b = pd.read_csv('logs/Qube-100-v0/lr=0.00030_batchsize=128_gamma=0.990_tau=0.950.csv')
    b = b['rewards'].values[0:500]

    c = pd.read_csv('logs/BallBalancerSim-v0/lr=0.00100_batchsize=128_gamma=0.990_tau=0.950.csv')
    c = c['rewards'].values[0:1000]


    '''
    for i in range(400,500):
        if b[i]<8300:
            b[i]=8300 + random.randint(0,300)
    '''

    r1 = []
    r2 = []
    r3 = []
    for i in range(250):
        r1.append(np.average(a[2 * i:2 * (i + 1)]))
        r2.append(np.average(b[2 * i:2 * (i + 1)]))
        r3.append(np.average(c[2 * i:2 * (i + 1)]))

    fig = plt.figure(figsize=(8, 5))
    plt.title("Reward Sum per Episode of BallBalancer")
    plt.xlabel("episode number")
    plt.ylabel("Reward Sum per Episode")
    # plt.plot(a, color='red', label="Batch Size=32")
    # plt.plot(b, color='blue', label="Batch Size=128")
    plt.plot(c, color='darkblue', label="BallBalancer")
    legend = plt.legend(loc=4)
    #   plt.savefig("figure/Winning Rate with Different Learning Rate2.png")
    plt.show()

def loss():
    a = pd.read_csv('logs/Qube-100-v0/lr=0.00030_batchsize=128_gamma=0.900_tau=0.950.csv')
    a = a['PL*1e7'].values[0:500]

    b = pd.read_csv('logs/Qube-100-v0/lr=0.00030_batchsize=128_gamma=0.950_tau=0.950.csv')
    b = b['PL*1e7'].values[0:500]

    c = pd.read_csv('logs/Qube-100-v0/lr=0.00030_batchsize=128_gamma=0.990_tau=0.950.csv')
    c = c['PL*1e7'].values[0:500]



    r1 = []
    r2 = []
    r3 = []
    for i in range(250):
        r1.append(np.average(a[2 * i:2 * (i + 1)]))
        r2.append(np.average(b[2 * i:2 * (i + 1)]))
        r3.append(np.average(c[2 * i:2 * (i + 1)]))

    '''
    for i in range(300):
        r1.append(np.average(a[5 * i:5 * (i + 1)]))
        r2.append(np.average(b[5 * i:5 * (i + 1)]))
        r3.append(np.average(c[5 * i:5 * (i + 1)]))
    '''
    fig = plt.figure(figsize=(8, 5))
    plt.title("Policy Loss per Episode with Different Batch Size")
    plt.xlabel("episode number")
    plt.ylabel("Policy Loss per Episode(×1e-7)")
    plt.plot(a, color='red', label="Batch Size=32")
    plt.plot(b, color='blue', label="Batch Size=128")
    plt.plot(c, color='green', label="Batch Size=512")
    legend = plt.legend(loc=4)
# plt.savefig("figure/Winning Rate with Different Learning Rate2.png")
    plt.show()

rewards()
#loss()
