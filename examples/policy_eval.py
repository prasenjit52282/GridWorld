import numpy as np
from gridenv import env
import matplotlib.pyplot as plt
from helper import inf_norm,getMRP
from PIL import Image

np.random.seed(42)
pi=np.random.choice(env.action_space,size=env.state_count) #random policy
image=Image.fromarray(env.getScreenshot(pi))
image.save(f"./logs/policy_eval/pi_selected.png")

#MDP to MRP
gamma=0.9
P_ss,R_s=getMRP(env,pi)

#Policy Evaluation
V=np.zeros((env.state_count,1))
V_prev=np.ones((env.state_count,1))

i=0
eps=1e-7
v_values=[]
while inf_norm(V-V_prev)>eps: #inf norm > eps
    V_prev=V.copy()
    V=R_s+gamma*np.matmul(P_ss,V_prev)
    v_values.append(inf_norm(V))
    i+=1
report=f"Converged in {i} iterations with eps {eps}\n"
report+=f"Pi= {pi}\n"
report+=f"V_pi= {V.flatten()}\n"
with open("./logs/policy_eval/report.txt","w") as f:f.write(report)
print(report)

plt.plot(v_values,lw=3,ls='--')
plt.ylabel('$|V|_{\infty}$',fontsize=16)
plt.xlabel('DP Iteration',fontsize=16)
plt.tight_layout()
plt.savefig("./logs/policy_eval/pi_eval_v.png")