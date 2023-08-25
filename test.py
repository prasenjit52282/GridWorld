# -*- coding: utf-8 -*-
from gridworld import GridWorld

import numpy as np

world=\
    """
    wwwwwwwwwwwwwwwww
    wa       w      w
    w     o  w      w
    wwwww    wwwwwwww
    w    o   wg     w
    wwwww    www  www
    w  o            w
    w   o  wwwwwwwwww
    w           o   w
    wwwwwwwwwwwwwwwww
    """
    
w=GridWorld(world,log=True)

policy=np.random.choice(w.action_space,size=w.state_count)

w.play_as_human(policy)