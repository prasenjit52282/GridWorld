import numpy as np

def inf_norm(vec):
    return np.max(np.abs(vec))

def getMRP(env,pi):
    P_ss=np.squeeze(np.take_along_axis(env.P_sas,pi.reshape(-1,1,1),axis=1),axis=1)
    R_s=np.take_along_axis(env.R_sa,pi.reshape(-1,1),axis=1)
    return P_ss,R_s