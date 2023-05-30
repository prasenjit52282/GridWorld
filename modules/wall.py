import pygame as pg

class Wall(pg.sprite.Sprite):
    def __init__(self,col,row):
        super().__init__()
        self.image=pg.transform.scale(pg.image.load('./images/wall.png'),(50,50))
        self.rect=self.image.get_rect()
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*50
        self.rect.y=self.pos.y*50

