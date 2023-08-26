import numpy as np

def inf_norm(vec):
    return np.max(np.abs(vec))

def l2_norm(vec):
    return np.sqrt(np.square(vec).sum())

def getMRP(env,pi):
    P_ss=np.squeeze(np.take_along_axis(env.P_sas,pi.reshape(-1,1,1),axis=1),axis=1)
    R_s=np.take_along_axis(env.R_sa,pi.reshape(-1,1),axis=1)
    return P_ss,R_s

def sample_trajectory(env,pi,gamma=0.9):
    tau=[]
    done=False
    total_reward=0
    s=env.reset()
    while not done:
        a=pi[s]
        s_,r,done,info=env.step(a)
        total_reward+=r
        tau.append([s,a,r,0])
        s=s_
    ret=0
    for idx in reversed(range(len(tau))):
        ret=tau[idx][2]+gamma*ret
        tau[idx][3]=ret
    return tau,total_reward

def eps_greedy_Q(eps,Q_sa,action_space):
    eps_greedy_actions=[]
    greedy_actions=np.argmax(Q_sa,axis=1)
    for i in range(len(greedy_actions)):
        if np.random.random()<eps:
            action=np.random.choice(action_space)
        else:
            action=greedy_actions[i]
        eps_greedy_actions.append(action)
    return eps_greedy_actions

def epsilon(curr_step,start=1,end=0.01,steady_step=300):
    if curr_step<steady_step:
        return 1+((end-start)/steady_step)*curr_step
    else:
        return end