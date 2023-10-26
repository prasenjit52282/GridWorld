import pygame as pg
import pkg_resources
from .block import Block

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
    
    def reInitilizeAgent(self):
        self.pos=pg.Vector2(self.initial_position.x,self.initial_position.y)
        self.set_pixcel_position()

    def setLoc(self,col,row):
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        
    def draw(self, screen):
       screen.blit(self.image, (self.rect.x, self.rect.y))