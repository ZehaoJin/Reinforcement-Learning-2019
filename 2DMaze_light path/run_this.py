# -*- coding: utf-8 -*-
"""
Created on Thu May 16 06:39:25 2019

@author: zehaojin
"""

"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable

reaching_episode=900
max_episode=1000

#'''
reaching_episode=9900
max_episode=10000
#'''

def update():
    for episode in range(max_episode):
        # initial observation
        time_counter = 0
        observation = env.reset()
        if episode > reaching_episode:
            sleeptime=0.1
        else:
            sleeptime=0
        while True:
            # fresh env
            env.render(sleeptime)

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done, time_counter = env.step(action, time_counter)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
        print('time used: ',time_counter)
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()