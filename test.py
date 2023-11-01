# -*- coding: utf-8 -*-
from gridworld import GridWorld

import numpy as np

world=\
    """
    wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    w  a                                     w
    w                                        w
    w                                 ooo    w
    w                               ooooooo  w
    w                              oooooooooow
    w                             ooooooooooow
    w                                        w
    wwwwwwwwwwwwwwwww                        w
    w                                        w
    w                                        w
    w                                        w
    wooo                                     w
    wooooooo                                 w
    woooooooooo                              w
    w oooooooooooo                           w
    w                                        w
    w                                        w
    w                                        w
    w                                        w
    w                   wwwwwwwwwwwwwwwwwwwwww
    w                                        w
    w                                        w
    w                                        w
    w                                        w
    w                                        w
    w                                        w
    w                                        w
    w                  ooo                   w
    w                ooooooo                 w
    w               oooooooooo               w
    w              oooooooooooo              w
    w                                        w
    w                                     gggw
    w                                   gggggw
    w                                 gggggggw
    w                               gggggggggw
    w                              ggggggggggw
    w                             gggggggggggw
    w                            ggggggggggggw
    wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    """
    
w=GridWorld(world,log=True,blocksize=(17,17),isDRL=True,viewsize=10,random_state=42)

policy=np.random.choice(w.action_values,size=w.state_count)

w.play_as_human(policy)