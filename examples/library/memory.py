import numpy as np
import tensorflow as tf

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

class Memory:
    def __init__(self,num_env=8,n_steps=512,epochs=10,steps_per_epoch=32,shuffle_buffer_size=1024):
        self.num_env=num_env
        self.n_steps=n_steps
        self.epochs=epochs
        self.steps_per_epoch=steps_per_epoch
        self.shuffle_buffer_size=shuffle_buffer_size
        self.batch_size=(self.num_env*self.n_steps)//self.steps_per_epoch

    def initilize(self,tensors):
        self.dataset=tf.data.Dataset.from_tensor_slices(tensors).\
                                     shuffle(self.shuffle_buffer_size).\
                                     repeat(self.epochs).\
                                     batch(self.batch_size,drop_remainder=True).\
                                     prefetch(tf.data.experimental.AUTOTUNE)