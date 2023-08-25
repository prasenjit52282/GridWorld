# -*- coding: utf-8 -*-
from gridworld import GridWorld

import numpy as np

world=\
    """
    wwwwwwwwwwwwwwwww
    wa         o    w
    w     o         w
    www      wwwwwwww
    w      o wg     w
    wwwww    ww   www
    w  o            w
    w        wwwwwwww
    w           o   w
    wwwwwwwwwwwwwwwww
    """
    
w=GridWorld(world,log=True)

policy=np.random.choice(w.action_space,size=w.state_count)

w.play_as_human(policy)