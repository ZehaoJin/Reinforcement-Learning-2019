# -*- coding: utf-8 -*-
"""
Created on Thu May 16 06:37:03 2019

@author: zehaojin
"""

"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example. The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE1_W, MAZE1_H = 3,3  #dimension of material 1
MAZE2_W, MAZE2_H = 3,3  #dimension of material 2
start_x, start_y = 1,1
end_x  , end_y   = 6,3
index_of_refraction1, index_of_refraction2 = 1, 3   #air, water=1,1.33 c/v

#'''
MAZE1_W, MAZE1_H = 10,20  #dimension of material 1
MAZE2_W, MAZE2_H = 10,20  #dimension of material 2
start_x, start_y = 1,1
end_x  , end_y   = 20,20
index_of_refraction1, index_of_refraction2 = 1, 3
#'''


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']   #up down left right
        self.n_actions = len(self.action_space)    #number of actions
        self.title('maze')
        self.geometry('{0}x{1}'.format((MAZE1_W+MAZE2_W) * UNIT, max(MAZE1_H,MAZE2_H) * UNIT))  #size of window, Width*Height in pixels
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=max(MAZE1_H,MAZE2_H) * UNIT,
                           width=(MAZE1_W+MAZE2_W) * UNIT)
        
        #create MAZE1
        self.MAZE1 = self.canvas.create_rectangle(
            0, 0,
            MAZE1_W*UNIT, MAZE1_H*UNIT, 
            fill='lightgrey')
        
        #create MAZE2
        self.MAZE2 = self.canvas.create_rectangle(
            MAZE1_W*UNIT, 0,
            (MAZE1_W+MAZE2_W)*UNIT, MAZE2_H*UNIT, 
            fill='dimgrey')
        
        
        # create grids
        for c in range(0, (MAZE1_W+MAZE2_W) * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, max(MAZE1_H,MAZE2_H) * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, max(MAZE1_H,MAZE2_H) * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, (MAZE1_W+MAZE2_W) * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT/2, UNIT/2]) #centre of a grid


        # create oval END point
        end_center = np.array([end_x * UNIT, end_y * UNIT]) - origin
        self.oval = self.canvas.create_oval(
            end_center[0] - (UNIT/2-UNIT/8), end_center[1] - (UNIT/2-UNIT/8),
            end_center[0] + (UNIT/2-UNIT/8), end_center[1] + (UNIT/2-UNIT/8),
            fill='yellow')
            
        # create oval START point
        start_center = np.array([start_x * UNIT, start_y * UNIT]) - origin
        self.oval_start = self.canvas.create_oval(
            start_center[0] - (UNIT/2-UNIT/8), start_center[1] - (UNIT/2-UNIT/8),
            start_center[0] + (UNIT/2-UNIT/8), start_center[1] + (UNIT/2-UNIT/8),
            fill='yellow')
            
        # create red rect (light)
        start_center = np.array([start_x * UNIT, start_y * UNIT]) - origin
        self.rect = self.canvas.create_rectangle(
            start_center[0] - (UNIT/2-UNIT/8), start_center[1] - (UNIT/2-UNIT/8),
            start_center[0] + (UNIT/2-UNIT/8), start_center[1] + (UNIT/2-UNIT/8),
            fill='red')

        # pack all
        self.canvas.pack()
        
        #print(self.canvas.coords(self.rect))

    def reset(self):
        self.update()
        #time.sleep(0.5) #pause
        self.canvas.delete(self.rect)
        origin = np.array([UNIT/2, UNIT/2])
        start_center = np.array([start_x * UNIT, start_y * UNIT]) - origin
        self.rect = self.canvas.create_rectangle(
            start_center[0] - (UNIT/2-UNIT/8), start_center[1] - (UNIT/2-UNIT/8),
            start_center[0] + (UNIT/2-UNIT/8), start_center[1] + (UNIT/2-UNIT/8),
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action,time_counter):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if (s[0] < ((MAZE1_W)*UNIT) and s[1] < ((MAZE1_H - 1) * UNIT)) or (s[0] > ((MAZE1_W)*UNIT) and s[1] < ((MAZE2_H - 1) * UNIT)):
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE1_W+MAZE2_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
                
        '''
        note: it lets the light to "stand still" if moving out of glasses, but it won't happen in real cases.plus, should
        the "standing still" consume time?
        '''

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        '''
        reward
        '''
        if base_action[0]!=0 or base_action[1]!=0:
            if s_[0] < (MAZE1_W)*UNIT:
                time_counter += index_of_refraction1
            else:
                time_counter += index_of_refraction2
        
        
        if s_ == self.canvas.coords(self.oval):
            reward = 1/time_counter    ##*(MAZE1_W+MAZE2_W)*max(MAZE1_H,MAZE2_H)
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False


        return s_, reward, done, time_counter

    def render(self,sleeptime):
        time.sleep(sleeptime)  #pause
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()