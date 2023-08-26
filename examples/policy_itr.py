import numpy as np
from gridenv import env
import matplotlib.pyplot as plt
from helper import inf_norm,getMRP
from PIL import Image

np.random.seed(42)

#MDP to MRP
gamma=0.9

#Policy Iteration
V=np.zeros((env.state_count,1))
pi=np.random.choice(env.action_space,size=env.state_count) #random policy
pi_prev=np.random.choice(env.action_space,size=env.state_count)

i=0
v_values=[]

while np.sum(np.abs(pi-pi_prev))>0:
    pi_prev=pi.copy()
    P_ss,R_s=getMRP(env,pi)
    V=R_s+gamma*np.matmul(P_ss,V)
    pi=np.argmax(env.R_sa+gamma*np.squeeze(np.matmul(env.P_sas,V)),axis=1)
    image=Image.fromarray(env.getScreenshot(pi))
    image.save(f"./logs/policy_itr/pi_{i}.png")
    v_values.append(inf_norm(V))
    i+=1

report=f"Converged in {i} iterations\n"
report+=f"Pi_*= {pi}\n"
report+=f"V_*= {V.flatten()}\n"
with open("./logs/policy_itr/report.txt","w") as f:f.write(report)
print(report)

plt.plot(v_values,lw=3,ls='--')
plt.ylabel('$|V|_{\infty}$',fontsize=16)
plt.xticks(range(len(v_values)),labels=["$\pi_{"+f"{e}"+"}$" for e in range(len(v_values))])
plt.xlabel('Policy',fontsize=16)
plt.tight_layout()
plt.savefig("./logs/policy_itr/pi_itr_v.png")