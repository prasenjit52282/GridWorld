import argparse
from library.constants import *
from library.npgalgo import NPG
from library.nn import ActorCritic
from library.mpe import SubprocVecEnv
from library.gridenv import big_renv_fn,make_env

# Instantiate the parser
parser = argparse.ArgumentParser(description='Training for different setting with NPG')
parser.add_argument('--random_state', type=int, default=random_state, help='Required random_state of env')
parser.add_argument('--exp_name', type=str, default="NPG", help='Required Experiment name to run')
parser.add_argument('--init_from_exp', type=str, default=None, help='Optional Experiment name to initilize weights from')
parser.add_argument('--log_loc', type=str, default="./logs/npg/", help='Require log_location for tensorboard')
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

	npg=NPG(envs,
            test_env,
            actor_critic_net,
            n_steps=n_steps_trpo,
            epochs=epochs_trpo,
            steps_per_epoch=steps_per_epoch_trpo,
            shuffle_buffer_size=shuffle_buffer_size_trpo,
            gamma=gamma,
            lam=lam,
            delta=delta,
            learning_rate=value_lr,
            cg_damping=cg_damping, 
            cg_iters=cg_iters, 
            residual_tol=residual_tol, 
            ent_coef=ent_coef,
            backtrack_coef=backtrack_coef,
            backtrack_iters=backtrack_iters_npg,
            cliprange_vf=cliprange_vf,
			log_loc=args.log_loc,
			only_test=args.test,
            log=False)

	if not args.test:
		npg.learn(train_for_step=train_for_step,test_at_iter=test_at_iter,num_of_test=num_of_test)
	else:
		stat=npg._test(num_of_test,args.render)
		print("Test performance: ",stat)
