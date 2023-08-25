import numpy as np
from gridenv import env
import matplotlib.pyplot as plt

np.random.seed(42)
pi=np.random.choice(env.action_space,size=env.state_count) #random policy
env.show(pi)

#MDP to MRP
gamma=0.9
P_ss=np.squeeze(np.take_along_axis(env.P_sas,pi.reshape(-1,1,1),axis=1),axis=1)
R_s=np.take_along_axis(env.R_sa,pi.reshape(-1,1),axis=1)

def inf_norm(vec):
    return np.max(np.abs(vec))

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
print(f"Converged in {i} iterations with eps {eps}")
print("V_pi=",V.flatten())
plt.plot(v_values,lw=3,ls='--')
plt.ylabel('$|V|_{\infty}$',fontsize=16)
plt.xlabel('DP Iteration',fontsize=16)
plt.tight_layout()
plt.savefig("./logs/Pi_eval.png")