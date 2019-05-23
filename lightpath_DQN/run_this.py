# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:07:50 2019

@author: zehaojin
"""

from maze_env import Maze
from RL_brain import DeepQNetwork

reaching_episode=290
max_episode=300
replace_target=200

#'''
reaching_episode=9900
max_episode=10000
replace_target=200
#'''

def run_maze():
    step = 0
    for episode in range(max_episode):
        # initial observation
        observation = env.reset()
        
        time_counter = 0
        if episode > reaching_episode:
            sleeptime=0.1
        else:
            sleeptime=0
            
        while True:
            # fresh env
            env.render(sleeptime)

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, time_counter = env.step(action,time_counter)

            RL.store_transition(observation, action, reward, observation_)

            if (step > replace_target) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
        print('time used: ',time_counter)

    # end of game
    print('game over')
    env.destroy()
    


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=replace_target,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()