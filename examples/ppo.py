import argparse
from library.constants import *
from library.ppoalgo import PPO2
from library.nn import ActorCritic
from library.mpe import SubprocVecEnv
from library.gridenv import big_renv_fn,make_env

# Instantiate the parser
parser = argparse.ArgumentParser(description='Training for different setting with PPO')
parser.add_argument('--random_state', type=int, default=random_state, help='Required random_state of env')
parser.add_argument('--exp_name', type=str, default="PPO", help='Required Experiment name to run')
parser.add_argument('--init_from_exp', type=str, default=None, help='Optional Experiment name to initilize weights from')
parser.add_argument('--log_loc', type=str, default="./logs/ppo/", help='Require log_location for tensorboard')
parser.add_argument('--test',action="store_true",help="to test pass this with init_from_exp tag")
parser.add_argument('--render',action="store_true",help="render while testing")

#Parse arguments
args = parser.parse_args()

if args.test and args.init_from_exp==None:raise Exception("Must provide --init_from_exp tag to restore model")

if __name__ == '__main__':
	envs=SubprocVecEnv([make_env(big_renv_fn,seed=args.random_state+i+1) for i in range(num_of_env)])
	test_env=make_env(big_renv_fn,seed=args.random_state)()

	actor_critic_net=ActorCritic(input_size=test_env.observation_space.shape[0],action_size=test_env.action_space.n,
								trainable=not args.test,exp_name=args.exp_name,init_from_exp=args.init_from_exp,seed=args.random_state)

	ppo=PPO2(envs,
			test_env,
			actor_critic_net,
			n_steps=n_steps,
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
			shuffle_buffer_size=shuffle_buffer_size,
			gamma=gamma,
			lam=lam,
			vf_coef=vf_coef,
			ent_coef=ent_coef,
			learning_rate=learning_rate,
			max_grad_norm=max_grad_norm,
			cliprange=cliprange,
			cliprange_vf=cliprange_vf,
			log_loc=args.log_loc)

	if not args.test:
		ppo.learn(train_for_step=train_for_step,test_at_iter=test_at_iter,num_of_test=num_of_test)
	else:
		stat=ppo._test(num_of_test,args.render)
		print("Test performance: ",stat)
