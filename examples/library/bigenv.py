from gridworld import GridWorld

big_world=\
    """
    wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    wa                                       w
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
    
env=GridWorld(big_world,slip=0.2,log=False,max_episode_step=2000,blocksize=(17,17),isDRL=True,viewsize=20,random_state=42)