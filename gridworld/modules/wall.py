import pygame as pg
import pkg_resources
from .block import Block

class Wall(pg.sprite.Sprite):
    def __init__(self,col,row):
        super().__init__()
        fpath=pkg_resources.resource_filename(__name__,'images/wall.png')
        self.image=pg.transform.scale(pg.image.load(fpath),Block.getBlockSize())
        self.rect=self.image.get_rect()
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*Block.sizeX
        self.rect.y=self.pos.y*Block.sizeY

