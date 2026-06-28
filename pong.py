
import gymnasium as gym
import ale_py
import math

def to_rad(angle_deg: float):
    return angle_deg * math.pi/180

POLE_ANGLE_MIN = to_rad(-24)
POLE_ANGLE_MAX = to_rad(24)

POLE_ANGLE_SPAN = abs(POLE_ANGLE_MIN) + abs(POLE_ANGLE_MAX)

def cartpole_reward(obs):
    cart_pos        = obs[0]; cart_vel        = obs[1]
    pole_angle      = obs[2]; pole_ang_vel    = obs[3]

    err_pole_angle = abs(pole_angle)
    reward_factor = 5
    reward = reward_factor * (1 - (err_pole_angle/POLE_ANGLE_MAX))
    return reward

# def action_generator():
    


env = gym.make("CartPole-v1", render_mode = 'human')

observation, info = env.reset(seed=42)
reward_cum = 0
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # reward shaping
    reward_cum += cartpole_reward(observation)

    print(observation)

    if terminated or truncated:
        observation, info = env.reset()
        reward_cum = 0

    # env.render()
env.close()