# -*- coding: utf-8 -*-
from gridworld import GridWorld

import numpy as np

world=\
    """
    wwwwwwwwwwwwwwwww
    wa       w      w
    w        w      w
    wwwww    wwwwwwww
    w        wg     w
    wwwww    www  www
    w               w
    w      wwwwwwwwww
    w               w
    wwwwwwwwwwwwwwwww
    """
    
w=GridWorld(world)

policy=np.random.choice(w.action_space,size=w.state_count)

w.setPolicy(policy)

w.play_as_human(True)