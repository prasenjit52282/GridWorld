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
        tau.append([s,a,r,0,0])
        s=s_
    #return computation
    ret=0
    for idx in reversed(range(len(tau))):
        ret=tau[idx][2]+gamma*ret
        tau[idx][3]=ret
    #unsafe tau marking
    if "hole" in info: #hole state at end
        for idx in (range(len(tau))):tau[idx][4]=1
    return tau,total_reward

def eps_greedy_Qsafe(eps,Q_sa,H_sa,unsafe_prob,action_space):
    eps_greedy_safe_actions=[]
    mask_safe=H_sa<=unsafe_prob
    #mask qs which have risk >unsafe_prob
    mask_all_unsafe=np.all(H_sa>unsafe_prob,axis=1,keepdims=True)
    #when all actions has >unsafe_prob risk take all qs for pi
    mask_eff=mask_safe+mask_all_unsafe
    Q_eff=np.multiply(Q_sa,mask_eff)
    eff_greedy_actions=np.argmax(Q_eff,axis=1)
    for i in range(len(eff_greedy_actions)):
        if np.random.random()<eps:
            safe_actions=list(filter(lambda e:mask_eff[i,e],action_space))
            action=np.random.choice(safe_actions)
        else:
            action=eff_greedy_actions[i]
        eps_greedy_safe_actions.append(action)
    return eps_greedy_safe_actions

def epsilon(curr_step,start=1,end=0.01,steady_step=300):
    if curr_step<steady_step:
        return 1+((end-start)/steady_step)*curr_step
    else:
        return end