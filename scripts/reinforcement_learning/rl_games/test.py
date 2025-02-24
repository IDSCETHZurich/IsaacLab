import random
import math
import matplotlib.pyplot as plt
import numpy as np
import itertools

n_players = 5
r = 1200 * np.ones(n_players, dtype=np.float32)
k = 10
P = np.random.uniform(0.9, 1.0, size=(n_players, n_players))

n_rounds = 1000
n_games_total = len(list(itertools.combinations(range(n_players), 2)))
r_list = np.zeros((n_players, n_rounds * n_games_total))
game_count = 0

for _ in range(n_rounds):
    for p1, p2 in itertools.combinations(range(n_players), 2):
        r_list[:, game_count] = r
        p = P[p1, p2] if p1 > p2 else 1 - P[p1, p2]
        s = random.uniform(0, 1) <= p
        r1 = r[p1] + k * (s - 1 / (1 + 10 ** ((r[p2] - r[p1]) / 400)))
        r2 = r[p2] + k * (1 - s - 1 / (1 + 10 ** ((r[p1] - r[p2]) / 400)))
        r[p1] = r1
        r[p2] = r2
        game_count += 1

""" for p1, p2 in itertools.combinations(range(n_players), 2):
    for _ in range(n_rounds):
        r_list[:, game_count] = r
        p = P[p1, p2] if p1 > p2 else 1 - P[p1, p2]
        s = random.uniform(0, 1) <= p
        r1 = r[p1] + k * (s - 1 / (1 + 10 ** ((r[p2] - r[p1]) / 400)))
        r2 = r[p2] + k * (1 - s - 1 / (1 + 10 ** ((r[p1] - r[p2]) / 400)))
        r[p1] = r1
        r[p2] = r2
        game_count += 1 """

for p in range(n_players):
    plt.plot(r_list[p], label=f"P{p}")
plt.legend()
plt.show()
