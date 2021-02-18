import numpy as np
import time
import gym
from quanser_robots.double_pendulum.ctrl import BalanceCtrl
import matplotlib.pyplot as plt


def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta


class PlotSignal:
    def __init__(self, window = 500):
        self.window = window
        self.values = {}

    def update(self, **argv):
        for k in argv:
            if k not in self.values:
                self.values[k] = [argv[k]]
            else:
                self.values[k].append(argv[k])
            self.values[k] = self.values[k][-self.window:]

    def plot_signal(self):
        N = len(self.values)
        plt.clf()
        for i, k in enumerate(self.values):

            plt.subplot(N, 1, i+1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.pause(0.0000001)

    def last_plot(self):
        N = len(self.values)
        plt.clf()
        plt.ioff()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i+1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.show()


def main():

    # Enable / Disable the plotting
    use_plot = False
    render = False

    if use_plot:
        plt.ion()
        window = 500
        real_plot = PlotSignal(window=window)

        collect_fr = 1  # Frequency collecting data
        plot_fr = 1  # Frequency refresh plot

    if render:
        render_fr = 10  # Render frequency: only for simulation

    # Initialize Environment:
    env = gym.make('DoublePendulumRR-v0')                     # Use "DoublePendulumRR-v0" for the simulation
    ctrl = BalanceCtrl(dt=env.env.timing.dt)

    print("\n\n###############################")
    print("Episode {0}".format(0))
    obs = env.reset()

    print("\nStart Controller:\t\t\t", end="")
    t_i = 0
    while not ctrl.done:

        t_i += 1
        act = ctrl(obs)
        obs, _, done, _ = env.step(act[0] * np.ones(1))

        if done and t_i > 100:
            break

        if render:
            if np.mod(t_i, render_fr) == 0:
                env.render()

        if use_plot:
            if np.mod(t_i, collect_fr) == 0:
                x, theta1, theta2, x_dot, theta1_dot, theta2_dot = obs
                real_plot.update(theta1=theta1, theta1_dot=theta1_dot, theta2=theta2, theta2_dot=theta2_dot,volt=act[0],u=act[1], x=x)

            if np.mode(t_i, plot_fr) == 0:
                real_plot.plot_signal()

    # Stop the cart:
    env.step(np.array([0.]))
    print("Finished!")

    # Close the Connection to the system:
    time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    main()
