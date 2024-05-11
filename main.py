import torch
from gym_anytrading.envs import StocksEnv

from agent import Agent
from rat.test_env import StockTradingEnv
import yfinance as yf

class Config:
    def __init__(self):
        self.gamma=0.95
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.episodes=200
        self.epsilon_start = 0.90
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon= 0.9
        self.learning_rate = 0.001
        self.memory_capacity = 1000
        self.replay_memory_size = 1000





def main():
    config = Config()
    data = yf.download("AAPL", period="5d", interval="1m")
    env=StockTradingEnv(data)
    env = StocksEnv(df=data, window_size=30, frame_bound=(30, len(data)), render_mode="human")
    batch_size=32
    # Example usage:
    # Initialize an agent
    state_size = 5  # Dimensionality of your state vector
    action_size = 3  # Number of possible actions
    agent = Agent(state_size, action_size,config)
    EPISODES=1000

    # Assuming you have some environment that provides state and reward
    # Example loop for interacting with the environment and training the agent:
    for episode in range(EPISODES):
        state = env.reset()  # Initialize state
        for time in range(500):  # Assuming max steps per episode is 500
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10  # Penalize if episode ends prematurely
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(episode, EPISODES, time, agent.epsilon))
                break
        agent.replay(batch_size)  # Train the agent using experience replay

if __name__ == '__main__':
    main()