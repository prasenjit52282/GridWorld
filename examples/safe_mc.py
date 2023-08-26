import time
import numpy as np
from gridenv import env
from helper import *
from tqdm import tqdm


#MC-Control
np.random.seed(42)

Q_sa=np.zeros((env.state_count,env.action_size))
N_sa=np.zeros((env.state_count,env.action_size))

gamma=0.99
episodes=100000
steady_explore_episode=20000
performance=[]

for episode in tqdm(range(episodes)):
    eps=epsilon(episode,start=1,end=0.01,steady_step=steady_explore_episode)
    pi=eps_greedy_Q(eps,Q_sa,env.action_space)    
    
    tau,total_reward=sample_trajectory(env,pi,gamma)
    performance.append(total_reward)
    
    seen=[]
    for s,a,r,G in tau:
        if (s,a) in seen:continue #first visit MC
        seen.append((s,a))
        
        N_sa[s,a]+=1
        Q_sa[s,a]+=(G-Q_sa[s,a])/(N_sa[s,a])

env.show(pi)

for e in range(10):
    done=False
    total_reward=0
    s=env.reset()
    while not done:
        a=pi[s]
        env.render()
        s_,r,done,info=env.step(a)
        total_reward+=r
        s=s_
        time.sleep(0.1)
    print(f"Episode {e} Total reward {total_reward}")
env.close()
