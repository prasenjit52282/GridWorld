import gym
import numpy as np
import pygame as pg
from gym.spaces import Box,Discrete
from collections import defaultdict
from .modules import Agent, Wall, Goal, State, Hole, Block

class GridWorld(gym.Env):
    def __init__(self,world,slip=0.2,log=False,max_episode_step=1000,blocksize=(50,50),isDRL=False,viewsize=10,random_state=None):
        super().__init__()
        Block.setBlockSize(*blocksize)
        self.isDRL=isDRL
        if self.isDRL:Block.setViewSize(viewsize)
        self.seed=random_state
        np.random.seed(random_state)

        self.world=world.split('\n    ')[1:-1]
        self.action_map={0:'right',1:'down',2:'left',3:'up'}
        self.action_values=[0,1,2,3]
        self.action_size=len(self.action_values)
        self.slip=slip
        self.logging=log
       
        self.col=len(self.world[0])
        self.row=len(self.world)
        self.state_color=(50,100,10)
        self.renderfirst=True
        self.policy={}
        self.episode_step=0
        self._max_epi_step=max_episode_step

        self.wall_group=pg.sprite.Group()
        self.state_group=pg.sprite.Group()
        self.state_dict=defaultdict(lambda :0)
        self.goal_group=pg.sprite.Group()

        i=0
        for y,et_row in enumerate(self.world):
            for x,block_type in enumerate(et_row):
                
                if block_type=='w':
                    self.wall_group.add(Wall(col=x,row=y))
                    
                elif block_type=='a':
                    self.agent=Agent(col=x,row=y,log=self.logging)
                    self.state_group.add(State(col=x,row=y,color=self.state_color))
                    self.state_dict[(x,y)]={'state':i,'reward':-1,'done':False,'type':'norm'}
                    i+=1
                    
                elif block_type=='g':
                    self.goal_group.add(Goal(col=x,row=y))
                    self.state_dict[(x,y)]={'state':i,'reward':100,'done':True,'type':'goal'}
                    i+=1

                elif block_type=='o':
                    self.state_group.add(Hole(col=x,row=y))
                    self.state_dict[(x,y)]={'state':i,'reward':-100,'done':True,"hole":True,'type':'hole'}
                    i+=1
                
                elif block_type==' ':
                    self.state_group.add(State(col=x,row=y,color=self.state_color))
                    self.state_dict[(x,y)]={'state':i,'reward':-1,'done':False,'type':'norm'}
                    i+=1
                    
        self.state_dict=dict(self.state_dict)
        self.state_count=len(self.state_dict)
        #setting action and observation space
        self.action_space=Discrete(self.action_size)
        if self.isDRL:
            self.observation_space=Box(low=-2,high=2,shape=(2*viewsize+1,2*viewsize+1),dtype='int8')
        else:
            self.observation_space=Discrete(self.state_count)
        #building environment model
        self.P_sas, self.R_sa=self.build_Model(self.slip)
        self.reset()


    def random_action(self):
        return np.random.choice(self.action_values)


    def formatState(self,response_state):
        if self.isDRL:
            return self.agent.getViewState(self.state_dict)
        else:
            return response_state
    

    def reset(self):
        self.episode_step=0
        self.agent.reInitilizeAgent()
        return self.formatState(self.state_dict[(self.agent.initial_position.x,self.agent.initial_position.y)]['state'])
    

    def get_action_with_probof_slip(self,action):
        individual_slip=self.slip/3
        prob=[individual_slip for a in self.action_values]
        prob[action]=1-self.slip
        act=np.random.choice(self.action_values,p=prob)
        return act

    
    def step(self,action,testing=False):
        if not testing:
            action=self.get_action_with_probof_slip(action)
        action=self.action_map[action]
        response=self.agent.move(action,self.wall_group,self.state_dict)
        self.episode_step+=1
        if "hole" in response:
            return self.formatState(response['state']),response['reward'],response['done'],{"hole":True}
        elif self.episode_step<=self._max_epi_step:
            return self.formatState(response['state']),response['reward'],response['done'],{}
        else:
            return self.formatState(response['state']),response['reward'],True,{'TimeLimit':True}
    

    def render(self):
        if self.renderfirst:
            pg.init()
            self.screen = pg.display.set_mode((self.col*Block.sizeX,self.row*Block.sizeY))
            self.renderfirst=False
        self.screen.fill(self.state_color)
        self.wall_group.draw(self.screen) 
        self.state_group.draw(self.screen)
        self.goal_group.draw(self.screen)    
        self.agent.draw(self.screen)
        pg.display.update()
        pg.display.flip()
        
    
    def close(self):
        self.renderfirst=True
        pg.quit()
        
        
    def __setPolicy(self,policy):
        for i,act in enumerate(policy):
            self.policy[i]=self.action_map[act]
        for s in self.state_group:
            s.change_with_policy(self.state_dict,self.policy)
    
    def __unsetPolicy(self):
        self.policy={}
        for s in self.state_group:
            s.default_state()
        
    def play_as_human(self,policy=None):
        if policy is not None:
            self.__setPolicy(policy)

        pg.init()
        screen = pg.display.set_mode((self.col*Block.sizeX,self.row*Block.sizeY))
        clock = pg.time.Clock()
        done = False
        while not done: 
            for event in pg.event.get():
                    if event.type == pg.QUIT:
                        done = True
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_LEFT:
                            response=self.agent.move('left',self.wall_group,self.state_dict)
                        elif event.key == pg.K_RIGHT:
                            response=self.agent.move('right',self.wall_group,self.state_dict)
                        elif event.key == pg.K_UP:
                            response=self.agent.move('up',self.wall_group,self.state_dict)
                        elif event.key == pg.K_DOWN:
                            response=self.agent.move('down',self.wall_group,self.state_dict)
                        if self.isDRL:self.formatState(response['state'])

            screen.fill(self.state_color)

            self.wall_group.draw(screen)  
            self.state_group.draw(screen)
            self.goal_group.draw(screen)    
            self.agent.draw(screen)
            
            pg.display.update()
            pg.display.flip()
            clock.tick(60)
        self.__unsetPolicy()
        pg.quit()

    def getScreenshot(self,policy=None):
        if policy is not None:
            self.__setPolicy(policy)
        pg.init()
        screen = pg.display.set_mode((self.col*Block.sizeX,self.row*Block.sizeY))
        screen.fill(self.state_color)

        self.wall_group.draw(screen)  
        self.state_group.draw(screen)
        self.goal_group.draw(screen)    
        self.agent.draw(screen)
        
        pg.display.update()
        pg.display.flip()
        image=pg.surfarray.array3d(screen).transpose(1,0,2)
        self.__unsetPolicy()
        pg.quit()
        return image

    
    def show(self,policy):
        self.play_as_human(policy)


    def build_Model(self,slip):
        P_sas=np.zeros((self.state_count,self.action_size,self.state_count),dtype="float32")
        R_sas=np.zeros((self.state_count,self.action_size,self.state_count),dtype="float32")

        for (col,row), curr_state in self.state_dict.items():
            for act in self.action_values:
                action=self.action_map[act]
                self.agent.setLoc(col,row)
                next_state=self.agent.move(action,self.wall_group,self.state_dict)
                P_sas[curr_state["state"],act,next_state["state"]]=1.0
                R_sas[curr_state["state"],act,next_state["state"]]=next_state["reward"]

        correct=1-slip
        ind_slip=slip/3
        for a in self.action_values:
            other_actions=[oa for oa in self.action_values if oa!=a]
            P_sas[:,a,:]=(P_sas[:,a,:]*correct)+(P_sas[:,other_actions,:].sum(axis=1)*ind_slip)

        R_sa=np.multiply(P_sas,R_sas).sum(axis=2)
        return P_sas,R_sa