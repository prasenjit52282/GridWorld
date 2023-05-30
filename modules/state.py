import pygame as pg

class State(pg.sprite.Sprite):
    def __init__(self,col,row):
        super().__init__()
        self.image=pg.Surface((50,50))
        self.rect=self.image.get_rect()
        self.pos=pg.Vector2(col,row)
        self.set_pixcel_position()
        
    def set_pixcel_position(self):
        self.rect.x=self.pos.x*50
        self.rect.y=self.pos.y*50
        
    def change_with_policy(self,state_dict,policy): #policy={0:'up',1:'down'} etc
        state=state_dict[(self.pos.x,self.pos.y)]['state']
        optimal_action=policy[state]
        self.image=pg.transform.scale(pg.image.load('./images/'+optimal_action+'.png'),(20,20))
