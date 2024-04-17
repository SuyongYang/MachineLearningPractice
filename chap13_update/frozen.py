import gym

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
observation = env.reset()

for _ in range(100):
  env.render()
  action = env.action_space.sample() 	# (1)
  observation, reward, done, info, _ = env.step(action)  # (2)

  if done:
    observation = env.reset()
env.close()