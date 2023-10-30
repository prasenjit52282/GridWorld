import numpy as np
from .constants import H
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import multiply,random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

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


class ActorCritic(Q_network):
	def __init__(self,input_size,action_size,trainable=True,exp_name=None,init_from_exp=None,seed=None):
		self.exp_name=exp_name
		self.trainable=trainable
		self.input_size=input_size
		self.action_size=action_size
		random.set_seed(seed)
		self.setup_nn(self.trainable,init_from_exp)

	@property
	def trainable_variables(self):
		return self.logits.trainable_variables+self.val.trainable_variables
	
	def setup_nn(self,trainable,init_from_exp=None):
		self.val=self.critic_network()
		if init_from_exp==None:
			self.logits=self.actor_network(self.action_size)
		else:
			self.logits=self.load_ppoModel(init_from_exp)
		self.val.trainable=trainable
		self.logits.trainable=trainable

	def Dense_network(self,name="ppo"):
		ann=Sequential(name=name)
		for i,h in enumerate(H):
			if i==0:ann.add(Dense(h,"relu",input_shape=(self.input_size,)))
			else:ann.add(Dense(h,"relu"))
		return ann

	def actor_network(self,action_size):
		actor=self.Dense_network('ppo_actor')
		actor.add(Dense(action_size,"linear"))
		return actor

	def critic_network(self):
		critic=self.Dense_network('ppo_critic')
		critic.add(Dense(1,"linear"))
		return critic

	def actor_head(self,s):
		logits=self.logits(s)
		dist=tfp.distributions.Categorical(logits=logits)
		return dist

	def critic_head(self,s):
		val=self.val(s)
		return val

	def action_log_prob_value(self,s):
		dist=self.actor_head(s)
		val=self.critic_head(s)
		a=dist.sample().numpy()
		#a,log_prob(s,a),v(s) --nograd
		return a,dist.log_prob(a).numpy().reshape(-1,1),val.numpy()

	def log_prob_value_entropy(self,s,a):
		dist=self.actor_head(s)
		val=self.critic_head(s)
		#log_prob(s,a),v(s),entropy(s) --withgrad
		return tf.reshape(dist.log_prob(a),shape=(-1,1)),val,dist.entropy()

	def value(self,s):
		val=self.critic_head(s)
		return val.numpy()

	def learned_action(self,s):
		logits=self.logits(np.array([s]))
		return tf.math.argmax(logits,axis=1).numpy()[0]

	def load_ppoModel(self,init_from_exp):
		return load_model(f"./model/{init_from_exp}_ppo.h5")

	def save_ppoModel(self):
		self.logits.save(f"./model/{self.exp_name}_ppo.h5")