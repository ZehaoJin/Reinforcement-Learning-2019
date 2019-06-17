# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:28 2019

@author: zehaojin
"""
import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None
    dt = 0.1    # refresh rate
    action_bound = [-1,1]
    goal = {'x': 250., 'z': 20., 'l': 20.}
    man = {'x': 250., 'z': 480., 'l': 20.} #500*500 window
    state_dim = 3
    action_dim = 2
    g=9.8
    m=1

    def __init__(self):
        self.man_info = np.array([self.man['x'],self.man['z']],dtype=np.float32)
        #self.t=0.
        #self.S=0.
        #self.done = False

    def step(self, action):
        window_x,window_z=500,1000
        self.done = False
        r = 0
        #print(action)
        action = np.clip(action, *self.action_bound)
        self.man_info += action
        if self.man_info[0] <= 0:
            self.man_info[0] = 0
        if self.man_info[0] >= window_x:
            self.man_info[0] = window_x
        if self.man_info[1] <= 0:
            self.man_info[1] = 0
        if self.man_info[0] >= window_z:
            self.man_info[0] = window_z
        
        #coords
        x, z = self.man_info
        dx, dz = action/self.dt
        #L and S
        L = 1/2*self.m*(dx**2+dz**2)-self.m*self.g*z
        dS = L*self.dt
        
        self.S += dS
        self.t += self.dt
        #distance
        #dist=(self.goal['x']-x)**2+(self.goal['z']-z)**2
        #r=-dist
     
        
        

        # done and reward
        if (self.goal['z']-10) <= z <=(self.goal['z']+10) and (self.goal['x']-10)<= x <=(self.goal['x']+10):
            self.done = True
            r = 1/(self.S)*100000

        # state
        #s = np.concatenate((self.man_info, np.array([self.t],dtype=np.float32), [1. if self.done else 0.]))
        #s = np.concatenate((self.man_info, action, np.array([self.t],dtype=np.float32)))
        s = np.concatenate((self.man_info, np.array([self.t],dtype=np.float32)))
        return s, r, self.done
 

    def reset(self):
        self.man_info[0]=self.man['x']
        self.man_info[1]=self.man['z']
        self.done=False
        self.t=0.
        self.S=0.
        #s = np.concatenate((self.man_info, np.array([self.t],dtype=np.float32), [1. if self.done else 0.]))
        #s = np.concatenate((self.man_info, np.array([0]),np.array([0]), np.array([self.t],dtype=np.float32)))
        s = np.concatenate((self.man_info, np.array([self.t],dtype=np.float32)))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.man_info, self.goal, self.man)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)*10-5


class Viewer(pyglet.window.Window):

    def __init__(self, man_info, goal, man):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=500, height=1000, resizable=False, caption='Free Fall', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.man_info = man_info

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['z'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['z'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['z'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['z'] - goal['l'] / 2]),
            ('c3B', (249, 86, 86) * 4))    # color
        self.man = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [man['x'] - man['l'] / 2, man['z'] - man['l'] / 2,                # location
                     man['x'] - man['l'] / 2, man['z'] + man['l'] / 2,
                     man['x'] + man['l'] / 2, man['z'] + man['l'] / 2,
                     man['x'] + man['l'] / 2, man['z'] - man['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))
            
            
    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        x = self.man_info[0]
        z = self.man_info[1]
        l = 20.
        
        self.man.vertices = [x - l/2, z - l/2,
                             x - l/2, z + l/2,
                             x + l/2, z + l/2,
                             x + l/2, z - l/2]
        


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        s = env.reset()
        for i in range(400):
            env.render()
            env.step(env.sample_action())
