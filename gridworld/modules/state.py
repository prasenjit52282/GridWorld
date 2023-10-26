import pygame as pg
import pkg_resources
from .block import Block

class State(pg.sprite.Sprite):
    def __init__(self,col,row,color):
        super().__init__()
        self.color=color
        self.default_state()
        self.rect=self.image.get_rect()
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()

    def default_state(self):
        self.image=pg.Surface(Block.getBlockSize())
        self.image.fill(self.color)
        
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*Block.sizeX
        self.rect.y=self.pos.y*Block.sizeY
        
    def change_with_policy(self,state_dict,policy): #policy={0:'up',1:'down'} etc
        state=state_dict[(self.pos.x,self.pos.y)]['state']
        optimal_action=policy[state]
        fpath=pkg_resources.resource_filename(__name__,'images/'+optimal_action+'.png')
        self.image=pg.transform.scale(pg.image.load(fpath),(int(Block.sizeX//2.5),int(Block.sizeY//2.5)))
