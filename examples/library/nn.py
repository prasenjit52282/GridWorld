from .constants import H
from tensorflow import multiply,random
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

class Q_network():
    def __init__(self,input_size,action_size,trainable=True,exp_name=None,init_from_exp=None,seed=None):
        self.input_size=input_size
        self.action_size=action_size
        self.trainable=trainable
        self.exp_name=exp_name
        random.set_seed(seed)
        self.nn=self.setup_nn(self.trainable,init_from_exp)

    def setup_nn(self,trainable,init_from_exp=None):
        if init_from_exp==None:
            nn=self.neuralNetArch()
        else:
            nn=self.load_dqnModel(init_from_exp)
        nn.trainable=trainable
        return nn

    def Dense_network(self,name="dqn"):
        ann=Sequential(name=name)
        for i,h in enumerate(H):
            if i==0:ann.add(Dense(h,"relu",input_shape=(self.input_size,)))
            else:ann.add(Dense(h,"relu"))
        ann.add(Dense(self.action_size,"linear"))
        return ann

    def neuralNetArch(self):
        ann=self.Dense_network()
        return ann

    def __call__(self,s,a_oneHot=None):
        q_val=self.nn(s)
        if a_oneHot is None:
            return q_val
        else:
            filtered_q_val=multiply(a_oneHot,q_val)
            return filtered_q_val

    def copy_from(self,other,tau=None):
        if tau is None:
            for tar,src in zip(self.nn.variables,other.nn.variables):
                tar.assign(src,read_value=False)
        else: #default=0.001
            for tar,src in zip(self.nn.variables,other.nn.variables):
                tar.assign(tau*src+(1-tau)*tar,read_value=False)

    def summary(self):
        self.nn.summary()

    def load_dqnModel(self,init_from_exp):
        return load_model(f"./model/{init_from_exp}_dqn.h5")

    def save_dqnModel(self):
        self.nn.save(f"./model/{self.exp_name}_dqn.h5")