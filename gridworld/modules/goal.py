import pygame as pg
import pkg_resources

class Goal(pg.sprite.Sprite):
    def __init__(self,col,row):
        super().__init__()
        fpath=pkg_resources.resource_filename(__name__,'images/goal.png')
        self.image=pg.transform.scale(pg.image.load(fpath),(50,50))
        self.rect=self.image.get_rect()
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*50
        self.rect.y=self.pos.y*50
        
    def draw(self, screen):
       screen.blit(self.image, (self.rect.x, self.rect.y))
