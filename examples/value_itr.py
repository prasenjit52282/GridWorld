import numpy as np
from gridenv import env
import matplotlib.pyplot as plt
from helper import inf_norm
from PIL import Image

np.random.seed(42)
gamma=0.9

#Value Iteration
pi=None
V=np.zeros((env.state_count,1))
V_prev=np.ones((env.state_count,1))

i=0
eps=1e-4
v_values=[]

while inf_norm(V-V_prev)>eps*((1-gamma)/(2*gamma)): #eps-optimality
    V_prev=V.copy()
    Q_sa=env.R_sa+gamma*np.squeeze(np.matmul(env.P_sas,V_prev))
    V=np.max(Q_sa,axis=1,keepdims=True)
    pi=np.argmax(Q_sa,axis=1)
    if i%10==0:
        image=Image.fromarray(env.getScreenshot(pi))
        image.save(f"./logs/value_itr/pi_{i}.png")
    v_values.append(inf_norm(V))
    i+=1

image=Image.fromarray(env.getScreenshot(pi))
image.save(f"./logs/value_itr/pi_{i}.png")

report=f"Converged to eps-optimal solution in {i} iterations with eps {eps}\n"
report+=f"Pi_*= {pi}\n"
report+=f"V_*= {V.flatten()}\n"
with open("./logs/value_itr/report.txt","w") as f:f.write(report)
print(report)

plt.plot(v_values,lw=3,ls='--')
plt.ylabel('$|V|_{\infty}$',fontsize=16)
plt.xticks(range(0,len(v_values),10),labels=["$\pi_{"+f"{e}"+"}$" for e in range(0,len(v_values),10)])
plt.xlabel('Policy',fontsize=16)
plt.tight_layout()
plt.savefig("./logs/value_itr/value_itr_v.png")