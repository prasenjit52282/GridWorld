import numpy as np
from .nn import Q_network
from .memory import replayBuffer
from tensorflow import GradientTape
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError

class dqn:
    def __init__(self,input_size,action_size,memory_size=100000,gamma=0.99,
                 exp_name=None,init_from_exp=None,seed=None):
        self.gamma=gamma
        self.memory_size=memory_size

        self.Q=Q_network(input_size,action_size,trainable=True,exp_name=exp_name,init_from_exp=init_from_exp,seed=seed)
        self.fixed_Q=Q_network(input_size,action_size,trainable=False,exp_name=exp_name,init_from_exp=init_from_exp,seed=seed)
        self.updateFixedQ()

        self.memory=replayBuffer(capacity=memory_size,obs_size=input_size)
        self.optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01)
        self.loss=MeanSquaredError()

    def getAction(self,s,epsilon=0.01):
        if np.random.random()<(1-epsilon):
            act=np.argmax(self.Q(np.array([s])),axis=1)[0]
        else:
            act=np.random.randint(0,self.Q.action_size)
        return act

    def learn(self,batch_size):
        s,a,r,s_,nd=self.memory.sample(batch_size)
        TD_target=r+self.gamma*np.multiply(nd,np.max(self.fixed_Q(s_),axis=1))
        a_oneHot=to_categorical(a,num_classes=self.Q.action_size)
        Q_target=np.multiply(TD_target.reshape(-1,1),a_oneHot)
        with GradientTape() as tape:
            loss=self.loss(Q_target,self.Q(s,a_oneHot))
        grads=tape.gradient(loss,self.Q.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.Q.nn.trainable_variables))
        return loss.numpy()

    def learnDDQN(self,batch_size):
        s,a,r,s_,nd=self.memory.sample(batch_size)
        a_anti_oneHot=to_categorical(np.argmax(self.Q(s_),axis=1),num_classes=self.Q.action_size)
        TD_target=r+self.gamma*np.multiply(nd,np.max(self.fixed_Q(s_,a_anti_oneHot),axis=1))
        a_oneHot=to_categorical(a,num_classes=self.Q.action_size)
        Q_target=np.multiply(TD_target.reshape(-1,1),a_oneHot)
        with GradientTape() as tape:
            loss=self.loss(Q_target,self.Q(s,a_oneHot))
        grads=tape.gradient(loss,self.Q.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.Q.nn.trainable_variables))
        return loss.numpy()

    def updateFixedQ(self):
        self.fixed_Q.copy_from(self.Q)

    def get_epsilon(self,step,steady_epsilon=0.01,steady_step=10000):
        if step>steady_step:
            return steady_epsilon
        else:
            m=(steady_epsilon-1)/steady_step
            return m*step+1

    def save(self):
        self.Q.save_dqnModel()