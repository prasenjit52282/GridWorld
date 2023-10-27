import numpy as np
from tqdm import tqdm
from gridenv import env
from helper import *
from PIL import Image
import matplotlib.pyplot as plt

#Safe SARSA
np.random.seed(42)

Q_sa=np.zeros((env.state_count,env.action_size))
H_sa=np.ones((env.state_count,env.action_size))
pi=np.random.choice(env.action_values,size=env.state_count) #random policy

gamma=0.99
gamma_risk=0.66 # 3 furure states
alpha=1e-3
unsafe_prob=0.1
train_steps=2000000
steady_explore_step=100000
performance=[]
risk=[]

done=False
s=env.reset()
total_reward=0
for step in tqdm(range(train_steps)):
    a=pi[s]
    s_p,r,done,info=env.step(a)
    total_reward+=r
    a_p=pi[s_p]
    Q_sa[s,a]+=alpha*((r+gamma*Q_sa[s_p,a_p])-Q_sa[s,a])
    if "hole" in info:
        H_sa[s,a]+=alpha*(1-H_sa[s,a])
    else:
        H_sa[s,a]+=alpha*((0+gamma_risk*H_sa[s_p].max())-H_sa[s,a])
    eps=epsilon(step,start=1,end=0.01,steady_step=steady_explore_step)
    pi[s]=online_safe_eps_greedy(eps,Q_sa[s],H_sa[s],unsafe_prob,env.action_values)
    s=s_p
    if done:
        performance.append(total_reward)
        risk.append(H_sa[0].mean())
        done=False
        s=env.reset()
        total_reward=0

#Greedy policy
pi=eps_greedy_Qsafe(0,Q_sa,H_sa,unsafe_prob,env.action_values)
image=Image.fromarray(env.getScreenshot(pi))
image.save(f"./logs/safe_sarsa/pi_emerged.png")

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
plt.savefig("./logs/safe_sarsa/reward.png")
plt.close()

plt.plot(risk,lw=2)
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel('Risk (0)',fontsize=16)
plt.xlabel('Episodes',fontsize=16)
plt.axhline(y=unsafe_prob,color='green',ls='-.',label='Unsafe threshold')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("./logs/safe_sarsa/risk.png")
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

with open("./logs/safe_sarsa/report.txt","w") as f:f.write(report)