# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:28 2019

@author: zehaojin
"""
import numpy as np
import pyglet

scale=10 ##just to enlarge the view.

'''
It is learning in a x*z=50*100 map
The agent "man" starts at (x,z)=25,48, with 0 initial velocity, and reaches near "goal" (25,2)  at som point.

'''


class Lightpath(object):
    viewer = None 
    action_bound = [-1,1]
    goal = {'x': 980., 'z': 250., 'l': 20.}
    man = {'x': 20., 'z': 250., 'l': 20.} #500*1000 display window
    state_dim = 6
    action_dim = 2
    index_of_refraction=1

    def __init__(self):
        self.man_info = np.array([self.man['x'],self.man['z']],dtype=np.float32)
        self.step_counter=0
        self.done = False
        self.t=0

    def step(self, action):
        
        self.man_info/=scale
        window_x,window_z=100,50
        self.done = False
        r = 0
        
        #take action
        action = np.clip(action, *self.action_bound)*10  #*3 so the bound is now [-3,3]
        
        ###print (dx,dz)
        #print(action)
        
        
        self.man_info += action
        #'''
        if self.man_info[0] <= 0:
            self.man_info[0] = 0
            action[0]=0
        if self.man_info[0] >= window_x:
            self.man_info[0] = window_x
            action[0]=0
        if self.man_info[1] <= 0:
            self.man_info[1] = 0
            action[1]=0
        if self.man_info[1] >= window_z:
            self.man_info[1] = window_z
            action[1]=0
        #'''
        #if self.man_info[0] <= 0 or self.man_info[0] >= window_x or self.man_info[1] <= 0 or self.man_info[1] >= window_z:
            #self.man_info -= action
            #dl=0
            #dt=0
        dx,dz=action
        x,z=self.man_info
        dl=np.sqrt(dx**2+dz**2)
        #v=c/n  ,with c=1, n=index of refraction
        #dt=dl/v
        dt=dl*self.index_of_refraction
        self.t+=dt
        self.step_counter+=1

      
        
        #r=-np.sqrt(((self.goal['x']/scale)-self.man_info[0])**2+((self.goal['z']/scale)-self.man_info[1])**2)
        r-=np.abs(self.man_info[0]-self.goal['x']/scale)
        #r-=np.abs(self.man_info[1]-self.goal['z']/scale)
        
        # done and reward
        #the if condition tells the localion of the goal! Current map is (x,z)=(100*50) and goal is (98+-10,25+-10)
        if self.man_info[0] >=(self.goal['x']/scale) and (self.goal['z']/scale-3) <= self.man_info[1] <=(self.goal['z']/scale+3):
        #if (self.goal['z']/scale-10) <= z <=(self.goal['z']/scale+10) and (self.goal['x']/scale-10)<= x <=(self.goal['x']/scale+10):
            self.done = True
            r = 1/self.t*50*100*100
        

        # state! current state is (x,z,step_counter,dl,dt,t)
        s = np.concatenate((self.man_info,np.array([self.step_counter]),np.array([dl]),np.array([dt]),np.array([self.t])))
        self.man_info*=scale
        #print(self.step_counter)
        return s, r, self.done
 

    def reset(self):
        self.man_info[0]=self.man['x']  
        self.man_info[1]=self.man['z']
        self.done=False
        self.t=0.
        self.step_counter=0
        self.man_info/=scale
        s = np.concatenate((self.man_info,np.array([self.step_counter]),np.array([0]),np.array([0]),np.array([self.t])))
        self.man_info*=scale
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.man_info, self.goal, self.man)
        self.viewer.render()

    def sample_action(self):
        return (np.random.rand(2)-0.5)*2


class Viewer(pyglet.window.Window):

    def __init__(self, man_info, goal, man):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=1000, height=500, resizable=False, caption='Lightpath', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.man_info = man_info

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        #https://www.rapidtables.com/web/color/RGB_Color.html
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
    env = Lightpath()
    while True:
        s = env.reset()
        for i in range(400):
            env.render()
            env.step(env.sample_action())
