import pygame as pg
from modules.agent import Agent
from modules.wall import Wall
from modules.goal import Goal
from modules.state import State
from collections import defaultdict
import numpy as np

class GridWorld:
    def __init__(self,world):
        
        self.world=world.split('\n    ')[1:-1]
        self.action_map={0:'right',1:'down',2:'left',3:'up'}
        self.action_space=[0,1,2,3]
        self.slip=0.2 #20% chance of taking wrong action
       
        self.col=len(self.world[0])    #10 num of columns in the above string
        self.row=len(self.world)       #5 num of rows in the above string
        self.state_color=(50,100,10)
        self.renderfirst=True
        self.policy={}
        self.episode_step=0
        self._max_epi_step=1000


        self.wall_group=pg.sprite.Group()
        self.state_group=pg.sprite.Group()

        self.state_dict=defaultdict(lambda :0)

        i=0
        for y,et_row in enumerate(self.world):
            for x,block_type in enumerate(et_row):
                
                if block_type=='w':
                    self.wall_group.add(Wall(col=x,row=y))
                    
                elif block_type=='a':
                    self.agent=Agent(col=x,row=y)
                    self.state_group.add(State(col=x,row=y))
                    self.state_dict[(x,y)]={'state':i,'reward':-1,'done':False}
                    i+=1
                    
                elif block_type=='g':
                    self.goal=Goal(col=x,row=y)
                    self.state_dict[(x,y)]={'state':i,'reward':10,'done':True}
                    i+=1
                
                elif block_type==' ':
                    self.state_group.add(State(col=x,row=y))
                    self.state_dict[(x,y)]={'state':i,'reward':-1,'done':False}
                    i+=1
                    
        self.state_dict=dict(self.state_dict)
        self.state_count=len(self.state_dict)
    
    
    def reset(self):
        self.episode_step=0
        self.agent.reInitilizeAgent()
        return self.state_dict[(self.agent.initial_position.x,self.agent.initial_position.y)]['state']
    
    def get_action_with_probof_slip(self,action): #slip property of env
        individual_slip=self.slip/3
        prob=[individual_slip for a in self.action_space]
        prob[action]=1-self.slip
        act=np.random.choice(self.action_space,p=prob)
        return act
        
    
    def step(self,action,testing=False):
        if not testing:
            action=self.get_action_with_probof_slip(action)
        action=self.action_map[action]
        response=self.agent.move(action,self.wall_group,self.state_dict)
        self.episode_step+=1
        
        if self.episode_step<=self._max_epi_step:
            return response['state'],response['reward'],response['done'],{}#info
        else:
            return response['state'],response['reward'],True,{'TimeLimit':True}
    
    def render(self):
        if self.renderfirst:
            pg.init()
            self.screen = pg.display.set_mode((self.col*50,self.row*50))
            
        self.screen.fill(self.state_color)
              
        self.wall_group.draw(self.screen)  
        self.goal.draw(self.screen)    
        self.agent.draw(self.screen)
            
        pg.display.update()
        pg.display.flip()
        
        
    
    def close(self):
        self.renderfirst=True
        pg.quit()
        
        
    def setPolicy(self,policy):
        for i,act in enumerate(policy):
            self.policy[i]=self.action_map[act]
            
        for s in self.state_group:
            s.change_with_policy(self.state_dict,self.policy)
        
        
    def play_as_human(self,show_policy=False):#policy={state_no:action('left')}
        if show_policy and len(self.policy)==0:
            raise Exception("Sorry, no policy found setPolicy first...use world.setPolicy([list of action for states])")

            
        pg.init()
        screen = pg.display.set_mode((self.col*50,self.row*50))

        clock = pg.time.Clock()

        done = False

        while not done:
            
            for event in pg.event.get():
                    if event.type == pg.QUIT:
                        done = True
         
                    elif event.type == pg.KEYDOWN:
                        if event.key == pg.K_LEFT:
                            response=self.agent.move('left',self.wall_group,self.state_dict)
                            #print(response)
                        elif event.key == pg.K_RIGHT:
                            response=self.agent.move('right',self.wall_group,self.state_dict)
                            #print(response)
                        elif event.key == pg.K_UP:
                            response=self.agent.move('up',self.wall_group,self.state_dict)
                            #print(response)
                        elif event.key == pg.K_DOWN:
                            response=self.agent.move('down',self.wall_group,self.state_dict) 
                            #print(response)
                        
                
            screen.fill(self.state_color)
              
            self.wall_group.draw(screen)  
            if show_policy:self.state_group.draw(screen)
            self.goal.draw(screen)    
            self.agent.draw(screen)
            
            pg.display.update()
            pg.display.flip()
            clock.tick(60)  
        pg.quit()
                
        
