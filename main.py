import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

# Create the environment
env = gym.make(
	"gym_cutting_stock/CuttingStock-v0",
	render_mode="human",  # Comment this line to disable rendering
)

if __name__ == "__main__":
	# Reset the environment
	observation, info = env.reset(seed=42)
	print(info)
	
	policy2210xxx = Policy2210xxx(1)
	while True:
		action = policy2210xxx.get_action(observation, info)
		observation, reward, terminated, truncated, info = env.step(action)
		print(info)

		if terminated or truncated:
		   observation, info = env.reset()

env.close()
