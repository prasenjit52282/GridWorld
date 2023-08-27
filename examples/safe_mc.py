import numpy as np
from gridenv import env
from helper import *
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


#Safe MC-Control
np.random.seed(42)

Q_sa=np.zeros((env.state_count,env.action_size))
H_sa=np.ones((env.state_count,env.action_size))

gamma=0.99
alpha=1e-4
beta=0.999
unsafe_prob=0.1
episodes=100000
steady_explore_episode=30000
performance=[]
risk=[]

for episode in tqdm(range(episodes)):
    eps=epsilon(episode,start=1,end=0.01,steady_step=steady_explore_episode)
    pi=eps_greedy_Qsafe(eps,Q_sa,H_sa,unsafe_prob,env.action_space)    
    
    tau,total_reward=sample_trajectory(env,pi,gamma)
    performance.append(total_reward)
    
    seen=[]
    for s,a,r,G,unsafe in tau:
        if (s,a) in seen:continue #first visit MC
        seen.append((s,a))
        
        Q_sa[s,a]+=alpha*(G-Q_sa[s,a])
        H_sa[s,a]=(beta*H_sa[s,a]+(1-beta)*unsafe)

    risk.append(H_sa[0].mean())

#Greedy policy
pi=eps_greedy_Qsafe(0,Q_sa,H_sa,unsafe_prob,env.action_space)
image=Image.fromarray(env.getScreenshot(pi))
image.save(f"./logs/safe_mc/pi_emerged.png")

mean_perf=np.lib.stride_tricks.sliding_window_view(performance,500).mean(axis=1)
std_perf=np.lib.stride_tricks.sliding_window_view(performance,500).std(axis=1)
plt.plot(mean_perf)
plt.fill_between(range(len(mean_perf)),mean_perf-std_perf,np.clip(mean_perf+std_perf,0,100),alpha=0.3)
plt.ylabel('Episode Reward',fontsize=16)
plt.xlabel('Episodes',fontsize=16)
plt.axhline(y=100,color='k',ls='--',label="Goal state Reward")
plt.legend(loc="lower right")
plt.grid(axis='x')
plt.tight_layout()
plt.savefig("./logs/safe_mc/reward.png")
plt.close()

plt.plot(risk,lw=2)
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel('Risk (0)',fontsize=16)
plt.xlabel('Episodes',fontsize=16)
plt.axhline(y=unsafe_prob,color='green',ls='-.',label='Unsafe threshold')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("./logs/safe_mc/risk.png")
plt.close()

report=""
pi_report=f"Pi= {pi}\n\nTest runs:"
report+=pi_report+"\n"
print(pi_report)
for e in range(10):
    done=False
    total_reward=0
    s=env.reset()
    while not done:
        a=pi[s]
        #env.render()
        s_,r,done,info=env.step(a)
        total_reward+=r
        s=s_
    epi_report=f"Episode {e} Total reward {total_reward}"
    report+=epi_report+"\n"
    print(epi_report)
env.close()

with open("./logs/safe_mc/report.txt","w") as f:f.write(report)