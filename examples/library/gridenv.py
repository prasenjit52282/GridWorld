from gridworld import GridWorld

small_world=\
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

small_env_fn=lambda seed:GridWorld(small_world,slip=0.2,log=False,max_episode_step=1000,seed=seed)
big_env_fn=lambda seed:GridWorld(big_world,slip=0.2,log=False,max_episode_step=2000,blocksize=(17,17),isDRL=True,viewsize=10,random_state=seed)


def make_env(env_fn,seed=0):
    def _init():
        env=env_fn(seed=seed)
        return env
    return _init


def dqn_test(test_env,num=1,steady_eps=None,agent=None):
    cum_rewd=0
    for _ in range(num):
        done=False
        rewd=0
        curr_state=test_env.reset()
        while not done:
            act=agent.getAction(curr_state,steady_eps)
            next_state,r,done,info=test_env.step(act)
            curr_state=next_state
            rewd+=r
        cum_rewd+=rewd
    test_env.close()
    avg_rewd=cum_rewd/num
    return {"avg_reward":avg_rewd}

def ppo_test(test_env,num=10,agent=None):
    cum_rewd=0
    for _ in range(num):
        done=False
        rewd=0
        curr_state=test_env.reset()
        while not done:
            act,_=agent.predict(curr_state)
            next_state,r,done,info=test_env.step(act)
            curr_state=next_state
            rewd+=r
        cum_rewd+=rewd
    test_env.close()
    avg_rewd=cum_rewd/num
    return {"avg_reward":avg_rewd}