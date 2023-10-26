import pygame as pg
import pkg_resources
from .block import Block

class Hole(pg.sprite.Sprite):
    def __init__(self,col,row):
        super().__init__()
        fpath=pkg_resources.resource_filename(__name__,'images/hole.png')
        self.image=pg.transform.scale(pg.image.load(fpath),Block.getBlockSize())
        self.rect=self.image.get_rect()
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        self.isHole=True
        
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*Block.sizeX
        self.rect.y=self.pos.y*Block.sizeY
        
    def draw(self, screen):
       screen.blit(self.image, (self.rect.x, self.rect.y))

    def change_with_policy(self,state_dict,policy): #policy={0:'up',1:'down'} etc
        pass
    
    def default_state(self):
        pass