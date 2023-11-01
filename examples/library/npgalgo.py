from copy import deepcopy
import time
import numpy as np
import tensorflow as tf
from .helper import *
from .memory import Memory
from .logger import TensorboardLogger

class NPG:
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
                 delta = 0.01,
                 learning_rate=0.00025,
                 cg_damping=0.001, 
                 cg_iters=10, 
                 residual_tol=1e-5, 
                 ent_coef=0.01,
                 backtrack_coef=0.6, 
                 backtrack_iters=1,
                 cliprange_vf=None,
                 log_loc=None,
                 only_test=False,
                 log=False):

        self.envs=envs
        self.test_env=test_env
        self.ac_network=actor_critic
        self.temp_ac_network=deepcopy(self.ac_network)

        self.n_steps=n_steps
        self.epochs=epochs
        self.num_envs=self.envs.num_envs
        self.steps_per_epoch=steps_per_epoch
        self.shuffle_buffer_size=shuffle_buffer_size
        self.mem=Memory(self.num_envs,self.n_steps,self.epochs,self.steps_per_epoch,self.shuffle_buffer_size)

        self.gamma=gamma
        self.lam=lam
        self.delta=delta
        self.ent_coef=ent_coef
        self.backtrack_coef=backtrack_coef
        self.backtrack_iters=backtrack_iters
        self.learning_rate=learning_rate
        self.cg_damping=cg_damping
        self.cg_iters=cg_iters
        self.residual_tol=residual_tol
        self.cliprange_vf=cliprange_vf

        self.optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        if only_test:self.logger=None
        else:self.logger=TensorboardLogger(loc=log_loc,experiment="NPG")
        self.log=log

    def policy_cost(self,new_log_probs,old_log_probs,adv):
        surr1=tf.reduce_mean(tf.multiply(new_log_probs, adv))
        return surr1 #maximize this

    def value_loss(self,states,old_values,returns):
        vpred=self.ac_network.value_with_grad(states)
        if self.cliprange_vf==None:
            return 0.5*tf.reduce_mean(tf.math.squared_difference(returns, vpred))
        else:
            vpred_clipped=old_values+tf.clip_by_value(vpred-old_values,-self.cliprange_vf, self.cliprange_vf)
            vf_losses1=tf.math.squared_difference(returns,vpred)
            vf_losses2=tf.math.squared_difference(returns,vpred_clipped)
            return 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    def entropy_loss(self,entropy):
        return -1*tf.reduce_mean(entropy)

    def surrogate_loss(self,ac_model,states,actions,old_log_probs,adv):
        new_log_probs,entropy=ac_model.log_prob_entropy(states,actions)
        return self.policy_cost(new_log_probs,old_log_probs,adv)+self.ent_coef*self.entropy_loss(entropy)

    def D_kl(self,ac_model,states,actions,old_log_probs):
        new_log_probs=ac_model.log_prob(states,actions)
        new_action_prob = tf.math.exp(new_log_probs) + 1e-8
        old_action_prob = tf.math.exp(old_log_probs)
        return tf.reduce_mean(tf.reduce_sum(old_action_prob * tf.math.log(old_action_prob / new_action_prob), axis=1))

    def hessian_vector_product(self,p,params):
        def hvp_fn(params):
            kl_grad_vector = flatgrad(self.D_kl, params, self.ac_network.logits.trainable_variables)
            grad_vector_product = tf.reduce_sum(kl_grad_vector * p)
            return grad_vector_product

        fisher_vector_product = flatgrad(hvp_fn, (params,), self.ac_network.logits.trainable_variables).numpy()
        return fisher_vector_product + (self.cg_damping * p)


    def _train(self,states,actions,old_log_probs,adv,old_values,returns):
        with tf.GradientTape() as t:
            v_loss=self.value_loss(states,old_values,returns)
        #value network update
        vgrads=t.gradient(v_loss,self.ac_network.val.trainable_variables)
        self.optimizer.apply_gradients(zip(vgrads,self.ac_network.val.trainable_variables))
        
        #policy network update
        surr_loss=self.surrogate_loss(self.ac_network,states,actions,old_log_probs,adv) # just to log
        policy_gradient = flatgrad(self.surrogate_loss,(self.ac_network,states,actions,old_log_probs,adv),self.ac_network.logits.trainable_variables).numpy()
        step_direction = conjugate_grad(self.hessian_vector_product, policy_gradient,self.cg_iters,self.residual_tol,params=(self.ac_network,states,actions,old_log_probs))
        shs = .5 * step_direction.dot(self.hessian_vector_product(step_direction,(self.ac_network,states,actions,old_log_probs)).T)

        lm = np.sqrt(shs / self.delta) + 1e-8
        fullstep = step_direction / lm
        if np.isnan(fullstep).any() and self.log:
            print("fullstep is nan")
            print("lm", lm)
            print("step_direction", step_direction)
            print("policy_gradient", policy_gradient)
        
        oldtheta = flatvars(self.ac_network.logits).numpy()
        #for one time update do backtrack_iters=1
        theta=linesearch(oldtheta, fullstep,self.temp_ac_network,states,actions,old_log_probs,adv,self.surrogate_loss,self.D_kl,self.backtrack_coef,self.backtrack_iters,self.delta,self.log)

        if np.isnan(theta).any() and self.log:print("NaN detected. Skipping update...")
        else:self.ac_network.assign_theta(theta)

        return surr_loss.numpy(),v_loss.numpy()

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

            returns      = functools_reduce_iconcat(returns).reshape(-1,1)
            log_probs    = functools_reduce_iconcat(log_probs)
            values       = functools_reduce_iconcat(values).reshape(-1,1)
            states       = functools_reduce_iconcat(states)
            actions      = functools_reduce_iconcat(actions)
            advantage    = returns - values
            advantage    = normalize(advantage)
            
            with self.logger.summary_writer.as_default():
                tf.summary.histogram("log_prob",log_probs,step=iteration)

            self.mem.initilize((states,actions,log_probs,advantage,values,returns))

            print("Training...")
            surr_loss,v_loss=[],[]
            for batch in self.mem.dataset:

                b_surr_loss,b_v_loss=self._train(*batch)
                surr_loss.append(b_surr_loss)
                v_loss.append(b_v_loss)

            loss_metrics={"surr_loss":np.mean(surr_loss),"value_loss":np.mean(v_loss)}
            self.logger.log(global_step,loss_metrics)
            
            if iteration%test_at_iter==0:
                score_dict=self._test(num_of_test)
                self.logger.log(global_step,score_dict)
                current_avg_reward=score_dict["avg_reward"]
                if current_avg_reward>max_reward:
                    max_reward=current_avg_reward
                    self.ac_network.save_ppoModel()
                print('Global step {} On iteration {} reward {}'.format(global_step,iteration,current_avg_reward))