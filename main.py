import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
import time

# Create the environment
env = gym.make(
	"gym_cutting_stock/CuttingStock-v0",
	render_mode="human",  # Comment this line to disable rendering
)

if __name__ == "__main__":
	# Reset the environment
	observation, info = env.reset(seed=None)
	print(info)

	NUM_TRIALS = 10
	trials = 0

	results = []
	elapsed_times = []

	clock = time.time()

	policy2210xxx = Policy2210xxx(2)
	while True:
		action = policy2210xxx.get_action(observation, info)
		observation, reward, terminated, truncated, info = env.step(action)
		print(info)

		if terminated or truncated:
			elapsed_times.append(time.time() - clock)

			results.append(info["trim_loss"])
			ave = sum(results)/len(results)
			ave_time = sum(elapsed_times) / len(elapsed_times)
			
			print("Trim losses: ", results)
			print("Average trim loss so far: ", ave)
			print("Average fill rate so far: ", 1.0 - ave)
			print("Average time: ", ave_time)

			trials += 1
			if trials >= NUM_TRIALS:
				break
			
			observation, info = env.reset()
			
			clock = time.time()

env.close()
