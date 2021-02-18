from gym.envs.registration import register

register(
    id='DoublePendulum-v0',
    entry_point='quanser_robots.double_pendulum.double_pendulum:DoublePendulum',
    max_episode_steps=10000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0}
)

register(
    id='DoublePendulumRR-v0',
    entry_point='quanser_robots.double_pendulum.double_pendulum_rr:DoublePendulum',
    max_episode_steps=10000,
    kwargs={'ip': '130.83.164.56', 'fs_ctrl': 500.0}
)
