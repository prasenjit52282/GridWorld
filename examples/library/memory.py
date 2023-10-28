import numpy as np
class replayBuffer:
    def __init__(self,capacity,obs_size=None):
        self.size=0
        self.looped=False
        self.capacity=capacity
        #pre-allocate memory for the buffer (reduce bottleneck due to replay buffer)
        self.s=np.zeros((self.capacity,obs_size),dtype="int8")
        self.a=np.zeros((self.capacity,),dtype="int8")
        self.r=np.zeros((self.capacity,),dtype="float32")
        self.s_=np.zeros((self.capacity,obs_size),dtype="int8")
        self.nd=np.zeros((self.capacity,),dtype="int8")

    def push(self,cs,a,r,ns,nd):
        self.s[self.size,:]=cs
        self.a[self.size]=a
        self.r[self.size]=r
        self.s_[self.size,:]=ns
        self.nd[self.size]=nd
        #keep track of the size
        self.size+=1
        if self.size==self.capacity:
            self.size%=self.capacity
            self.looped=True

    def sample(self,batch_size):
        if self.looped:
            #if buffer is full sample from the whole experience
            batch_index=np.random.randint(0,self.capacity,batch_size)
        else:
            #else sample from the valid set of experiences
            assert self.size>=10000,'Not enough samples for batch update....please put at least 10000 samples'
            batch_index=np.random.randint(0,self.size,batch_size)
        #return the batch of experiences
        return np.float32(self.s[batch_index]),self.a[batch_index],self.r[batch_index],np.float32(self.s_[batch_index]),self.nd[batch_index]