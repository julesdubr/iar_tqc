from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy

# Load model and env
model = TQC.load("tqc_humanoid")
env = model.get_env()

# Evaluate the model
# mean_reward, std_rewards = evaluate_policy(model, env, n_eval_episodes=100)

# Render
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
