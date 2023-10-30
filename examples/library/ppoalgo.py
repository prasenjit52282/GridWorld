import time
import numpy as np
import tensorflow as tf
from .helper import *
from .memory import Memory
from .logger import TensorboardLogger

class PPO2:
	def __init__(self,
				 envs,
				 test_env,
				 actor_critic,
				 n_steps=512,
				 epochs=10,
				 steps_per_epoch=32,
				 shuffle_buffer_size=1024,
				 gamma=0.99,
				 lam=0.95,
				 vf_coef=0.5,
				 ent_coef=0.01,
				 learning_rate=0.00025,
				 max_grad_norm=0.5,
				 cliprange=0.2,
				 cliprange_vf=None,
				 log_loc=None
				 ):

		self.envs=envs
		self.test_env=test_env
		self.ac_network=actor_critic

		self.n_steps=n_steps
		self.epochs=epochs
		self.num_envs=self.envs.num_envs
		self.steps_per_epoch=steps_per_epoch
		self.shuffle_buffer_size=shuffle_buffer_size
		self.mem=Memory(self.num_envs,self.n_steps,self.epochs,self.steps_per_epoch,self.shuffle_buffer_size)

		self.gamma=gamma
		self.lam=lam
		self.vf_coef=vf_coef
		self.ent_coef=ent_coef
		self.learning_rate=learning_rate
		self.max_grad_norm=max_grad_norm
		self.cliprange=cliprange
		self.cliprange_vf=cliprange_vf

		self.optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
		self.logger=TensorboardLogger(loc=log_loc,experiment="PPO")

	def policy_loss(self,new_log_probs,old_log_probs,adv):
		r=tf.math.exp(tf.math.subtract(new_log_probs, old_log_probs))
		surr1=tf.multiply(r, adv)
		surr2=tf.multiply(tf.clip_by_value(r, 1-self.cliprange, 1+self.cliprange), adv)
		loss= -1*tf.reduce_mean(tf.reduce_min([surr1,surr2],axis=0))
		return loss

	def value_loss(self,vpred,old_values,returns):
		if self.cliprange_vf==None:
			return 0.5*tf.reduce_mean(tf.math.squared_difference(returns, vpred))
		else:
			vpred_clipped=old_values+tf.clip_by_value(vpred-old_values,-self.cliprange_vf, self.cliprange_vf)
			vf_losses1=tf.math.squared_difference(returns,vpred)
			vf_losses2=tf.math.squared_difference(returns,vpred_clipped)
			return 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

	def entropy_loss(self,entropy):
		return -1*tf.reduce_mean(entropy)


	def _train(self,states,actions,old_log_probs,adv,old_values,returns):
		with tf.GradientTape() as t:
			new_log_probs,vpred,entropy=self.ac_network.log_prob_value_entropy(states,actions)
			pi_loss=self.policy_loss(new_log_probs,old_log_probs,adv)
			v_loss=self.vf_coef*self.value_loss(vpred,old_values,returns)
			ent_loss=self.ent_coef*self.entropy_loss(entropy)
			total_loss=pi_loss+v_loss+ent_loss

		grads=t.gradient(total_loss,self.ac_network.trainable_variables)
		if self.max_grad_norm!=None:
			grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
		self.optimizer.apply_gradients(zip(grads,self.ac_network.trainable_variables))
		return pi_loss.numpy(),v_loss.numpy(),ent_loss.numpy()

	def _test(self,num=1,render=False):
		cum_rewd=0
		for _ in range(num):
			done=False
			rewd=0
			curr_state=self.test_env.reset()
			while not done:
				if render:
					self.test_env.render()
					time.sleep(0.02)
				act=self.ac_network.learned_action(curr_state)
				next_state,r,done,info=self.test_env.step(act)
				curr_state=next_state
				rewd+=r
			cum_rewd+=rewd
		self.test_env.close()
		avg_rewd=cum_rewd/num
		return {"avg_reward":avg_rewd}


	def learn(self,train_for_step=10000,test_at_iter=2,num_of_test=10):
		global_step=0
		max_reward=-np.inf
		iterations=train_for_step//(self.n_steps*self.num_envs)
		for iteration in range(iterations):
			print("Collecting...")
			log_probs = []
			values = []
			states = []
			actions = []
			rewards = []
			masks = []

			curr_states=self.envs.reset()
			for _ in range(self.n_steps):
				act,log_prob,val=self.ac_network.action_log_prob_value(curr_states)
				next_states,rewds,dones,infos=self.envs.step(act)

				log_probs.append(log_prob)
				values.append(val.ravel())
				states.append(curr_states)
				actions.append(act)
				rewards.append(rewds)
				masks.append(1-dones)

				curr_states=next_states
				global_step+=self.num_envs

			next_value=self.ac_network.value(next_states)
			returns=compute_gae(next_value.ravel(), rewards, masks, values,self.gamma,self.lam)

			returns	  = functools_reduce_iconcat(returns).reshape(-1,1)
			log_probs	= functools_reduce_iconcat(log_probs)
			values	   = functools_reduce_iconcat(values).reshape(-1,1)
			states	   = functools_reduce_iconcat(states)
			actions	  = functools_reduce_iconcat(actions)
			advantage	= returns - values
			advantage	= normalize(advantage)
			
			with self.logger.summary_writer.as_default():
				tf.summary.histogram("log_prob",log_probs,step=iteration)

			self.mem.initilize((states,actions,log_probs,advantage,values,returns))

			print("Training...")
			pi_loss,v_loss,ent_loss=[],[],[]
			for batch in self.mem.dataset:

				b_pi_loss,b_v_loss,b_ent_loss=self._train(*batch)
				pi_loss.append(b_pi_loss)
				v_loss.append(b_v_loss)
				ent_loss.append(b_ent_loss)

			loss_metrics={"policy_loss":np.mean(pi_loss),"value_loss":np.mean(v_loss),"entropy_loss":np.mean(ent_loss)}
			self.logger.log(global_step,loss_metrics)
			
			if iteration%test_at_iter==0:
				score_dict=self._test(num_of_test)
				self.logger.log(global_step,score_dict)
				current_avg_reward=score_dict["avg_reward"]
				if current_avg_reward>max_reward:
					max_reward=current_avg_reward
					self.ac_network.save_ppoModel()
				print('Global step {} On iteration {} reward {}'.format(global_step,iteration,current_avg_reward))