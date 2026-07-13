import random

# Toy game to get familiar with Q-learning before making CR sim
class Gridworld:
    def __init__(self, size=5, max_steps=100):
        self.size = size
        self.max_steps = max_steps
        self.n_actions = 4

        self.x, self.y = 0, 0
        self.goal = (self.size - 1, self.size - 1)
        self.steps = 0

    def reset(self) -> int:
        self.x, self.y = 0, 0
        self.goal = (self.size - 1, self.size - 1)
        self.steps = 0
        return self._obs()

    def _obs(self) -> int:
        return self.y * self.size + self.x # 0 to size^2 - 1

    def step(self, action) -> tuple[int, float, bool]:
        self.steps += 1

        if action == 0 and self.y > 0: # up
            self.y -= 1
        elif action == 1 and self.x < self.size - 1: # right
            self.x += 1
        elif action == 2 and self.y < self.size - 1: # down
            self.y += 1
        elif action == 3 and self.x > 0: # left
            self.x -= 1

        done = False
        reward = -0.01

        if (self.x, self.y) == self.goal:
            reward = 1.0
            done = True
        if self.steps >= self.max_steps:
            done = True

        return self._obs(), reward, done

    def render(self):
        for yy in range(self.size):
            row = []
            for xx in range(self.size):
                if (xx, yy) == (self.x, self.y):
                    row.append('A')
                elif (xx, yy) == self.goal:
                    row.append('G')
                else:
                    row.append('.')
            print(''.join(row))
        print()

def q_learning(env: Gridworld, episodes=2000, alpha=0.2, gamma=0.99, eps=0.2):
    n_states = env.size * env.size
    Q = [[0.0] * env.n_actions for _ in range(n_states)] # Q[state][action]

    for ep in range(episodes):
        s = env.reset()
        done = False

        while not done:
            if random.random() < eps: # exploration
                a = random.randrange(env.n_actions)
            else: # exploitation
                a = max(range(env.n_actions), key=lambda action: Q[s][action])

            s2, r, done = env.step(a)
            td_error = r + gamma * max(Q[s2]) - Q[s][a] # temporal difference error
            Q[s][a] += alpha * td_error

            s = s2
    return Q

def run_policy(env: Gridworld, Q: list[list[float]]):
    s = env.reset()
    env.render()

    done = False

    while not done:
        a = max(range(env.n_actions), key=lambda action: Q[s][action])

        s, _, done = env.step(a)
        env.render()

grid = Gridworld(5, 50)
Q = q_learning(grid)
run_policy(grid, Q)
