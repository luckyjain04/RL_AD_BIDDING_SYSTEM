import numpy as np

class RTBEnv:
    def __init__(self, budget=100000):
        self.budget = float(budget)
        self.initial_budget = float(budget)
        self.state_dim = 5 
        # Bid Buckets in INR
        self.action_space = [0.0, 5.0, 20.0, 50.0, 100.0, 500.0] 
        self.reset()

    def reset(self):
        self.budget = self.initial_budget
        self.current_state = np.random.uniform(0, 1, self.state_dim).astype(np.float32)
        return self.current_state

    def step(self, action_idx):
        bid = float(self.action_space[action_idx])
        # Market price around ₹30-₹70
        market_price = float(np.random.lognormal(mean=3.5, sigma=0.5))
        won_auction = bid >= market_price
        reward = 0.0
        clicked = False

        if won_auction and self.budget >= bid:
            self.budget -= bid
            interest = float(self.current_state[0])
            click_prob = (interest * 0.45) + (bid * 0.001) 
            clicked = bool(np.random.rand() < click_prob)
            reward = 15.0 if clicked else -0.5
        elif won_auction and self.budget < bid:
            reward = -2.0
        
        self.current_state = np.random.uniform(0, 1, self.state_dim).astype(np.float32)
        done = bool(self.budget <= 0)
        return self.current_state, float(reward), done, {"clicked": clicked, "cost": float(bid if won_auction else 0)}