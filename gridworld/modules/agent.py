import numpy as np
import pygame as pg
import pkg_resources
from .block import Block
from itertools import product

class Agent(pg.sprite.Sprite):
    def __init__(self,col,row,log):
        super().__init__()
        self.log=log
        fpath=pkg_resources.resource_filename(__name__,'images/agent.png')
        self.image=pg.transform.scale(pg.image.load(fpath),Block.getBlockSize())
        self.rect=self.image.get_rect()
        self.initial_position=pg.Vector2(col,row)
        
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
    
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*Block.sizeX
        self.rect.y=self.pos.y*Block.sizeY
    
    def move(self,direction,walls,state_dict):
        pastpos=pg.Vector2(self.pos.x,self.pos.y)
        if hasattr(state_dict[(pastpos.x,pastpos.y)],"isHole"):
            self.pos=pg.Vector2(pastpos.x,pastpos.y)
        elif direction=='down':
            self.pos+=pg.Vector2(0,1)
        elif direction=='up':
            self.pos+=pg.Vector2(0,-1)
        elif direction=='right':
            self.pos+=pg.Vector2(1,0)
        elif direction=='left':
            self.pos+=pg.Vector2(-1,0)
        for wall in walls:
            if self.pos==wall.pos:
                self.pos=pg.Vector2(pastpos.x,pastpos.y)
                break
        self.set_pixcel_position()
        next_state=state_dict[(self.pos.x,self.pos.y)]
        if self.log:print(next_state)
        return next_state

    def get_state_symbol(self,Type):
        if Type=='agent':return 0 
        elif Type=='goal':return +2
        elif Type=='hole':return -2
        elif Type=='norm':return +1
        elif Type=='unknown':return -1
        
    def getViewState(self,state_dict):
        hh,hw=Block.getViewSize()
        s=np.full((2*hh+1,2*hw+1),fill_value=self.get_state_symbol('unknown'))
        s[hh-0,hw-0]=self.get_state_symbol("agent")
        for spread in range(1,hh+1):
            dir=list(product([-spread,spread],range(-spread,spread+1)))+list(product(range(-(spread-1),spread),[-spread,spread]))
            # print(dir)
            for mx,my in dir:
                try:
                    s[hh+(mx),hw+(my)]=self.get_state_symbol(state_dict[(self.pos.x+mx,self.pos.y+my)]["type"])
                except KeyError:
                    pass
        curr_state_type=self.get_state_symbol(state_dict[(self.pos.x,self.pos.y)]["type"])
        if np.abs(curr_state_type)==2: # terminal states (-2: hole, +2: goal)
            s[hh-0,hw-0]=curr_state_type
        state=np.rot90(np.fliplr(s))
        if self.log:
            print(state)
        return state
    
    def reInitilizeAgent(self):
        self.pos=pg.Vector2(self.initial_position.x,self.initial_position.y)
        self.set_pixcel_position()

    def setLoc(self,col,row):
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        
    def draw(self, screen):
       screen.blit(self.image, (self.rect.x, self.rect.y))