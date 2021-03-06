# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:28 2019

@author: zehaojin
"""
import numpy as np
import pyglet

scale=10 ##just to enlarge the view.

'''
It is learning in a x*z=100*50 map
The agent "man" starts at (x,z)=25,48, with 0 initial velocity, and reaches near "goal" (25,2)  at som point.

'''


class Lightpath(object):
    viewer = None 
    action_bound = [-1,1]
    goal = {'x': 980., 'z': 450., 'l': 20.}
    man = {'x': 20., 'z': 50., 'l': 20.} #1000*500 display window
    state_dim = 7
    action_dim = 2
    index_of_refraction1,index_of_refraction2=1,1.3

    def __init__(self):
        self.man_info = np.array([self.man['x'],self.man['z']],dtype=np.float32)
        self.step_counter=0
        self.done = False
        self.t=0
        self.best_t=133.8

    def step(self, action):
        ###500,280 as turning point
        self.man_info/=scale
        window_x,window_z=100,50
        self.done = False
        r = 0
        
        #take action
        action = np.clip(action, *self.action_bound)*0.5  #*3 so the bound is now [-3,3]
        action+=0.5
        ###print (dx,dz)
        #print(action)
        
        
        #self.man_info += action*np.array([100.0,50.0])
        self.man_info = np.array([50.0,action[1]*50.0],dtype=np.float32)
        #print(self.man_info)
        '''
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
        '''
        #if self.man_info[0] <= 0 or self.man_info[0] >= window_x or self.man_info[1] <= 0 or self.man_info[1] >= window_z:
            #self.man_info -= action
            #dl=0
            #dt=0
        x,z=self.man_info
        dx1=x-self.man['x']/scale
        dz1=z-self.man['z']/scale
        dx2=self.goal['x']/scale-x
        dz2=self.goal['z']/scale-z
        dl1=np.sqrt(dx1**2+dz1**2)
        dl2=np.sqrt(dx2**2+dz2**2)
        #v=c/n  ,with c=1, n=index of refraction
        #dt=dl/v
        dt1=dl1*self.index_of_refraction1
        dt2=dl2*self.index_of_refraction2

            
        self.t=dt1+dt2
        self.step_counter+=1

        print('loc:',self.man_info,'  t:',self.t)
        # done and reward
        #the if condition tells the localion of the goal! Current map is (x,z)=(100*50) and goal is (98+-10,25+-10)
        self.done = True
        ###r = 1/(self.t)**2*(50*100*100)**2
        r = (self.best_t-self.t)*1000*np.abs(self.best_t-self.t)
        #r = (self.best_t-self.t)*1000
        
        if self.t < self.best_t:
            self.best_t=self.t
        

        # state! current state is (x,z,step_counter,dl,dt,t)
        s = np.concatenate((self.man_info,np.array([dl1]),np.array([dl2]),np.array([dt1]),np.array([dt2]),np.array([self.t])))
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
        s = np.concatenate((self.man_info,np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([self.t])))
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
        self.leftmaterial = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [0, 0,                # location
                     0, 500,
                     500, 500,
                     500, 0]),
            ('c3B', (224, 224, 224) * 4))
        self.rightmaterial = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [500, 0,                # location
                     500, 500,
                     1000, 500,
                     1000, 0]),
            ('c3B', (160, 160, 160) * 4))
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
