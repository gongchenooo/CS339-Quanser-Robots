from gym.envs.registration import register


register(
    id='BallBalancerSim-v0',
    entry_point='quanser_robots.ball_balancer.ball_balancer_sim:BallBalancerSim',
    max_episode_steps=1000,
    kwargs={'fs': 250.0, 'fs_ctrl': 250.0, 'simplified_dyn': False}
)

register(
    id='BallBalancerSimSimpleDyn-v0',
    entry_point='quanser_robots.ball_balancer.ball_balancer_sim:BallBalancerSim',
    max_episode_steps=1000,
    kwargs={'fs': 250.0, 'fs_ctrl': 250.0, 'simplified_dyn': True}
)

register(
    id='BallBalancerRR-v0',
    entry_point='quanser_robots.ball_balancer.ball_balancer_rr:BallBalancerRR',
    max_episode_steps=5000,
    kwargs={'ip': '130.83.164.52', 'fs_ctrl': 500.0}
)

register(
    id='BallBalancerRR-v1',
    entry_point='quanser_robots.ball_balancer.ball_balancer_rr:BallBalancerRR',
    max_episode_steps=5000,
    kwargs={'ip': '130.83.164.52', 'fs_ctrl': 250.0}
)
