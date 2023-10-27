from gridworld import GridWorld

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

env=GridWorld(world,slip=0.2,log=False,max_episode_step=1000)