import argparse
import numpy as np
from tqdm import tqdm
from library.dqnalgo import dqn
from library.constants import *
from library.logger import TensorboardLogger
from library.gridenv import make_env,dqn_test,big_renv_fn

# Instantiate the parser
parser = argparse.ArgumentParser(description='Training for different setting with DQN')
parser.add_argument('--random_state', type=int, default=random_state, help='Required random_state of env')
parser.add_argument('--exp_name', type=str, default="DQN", help='Required Experiment name to run')
parser.add_argument('--init_from_exp', type=str, default=None, help='Optional Experiment name to initilize weights from')
parser.add_argument('--log_loc', type=str, default="./logs/dqn/", help='Require log_location for tensorboard')

#Parse arguments
args = parser.parse_args()


env=make_env(big_renv_fn,args.random_state)()
agent=dqn(input_size=env.observation_space.shape[0],action_size=env.action_space.n,memory_size=memory_size,
          gamma=gamma,exp_name=args.exp_name,init_from_exp=args.init_from_exp,seed=args.random_state)
logger=TensorboardLogger(loc=args.log_loc,experiment=args.exp_name)

print("Collecting random transitions .....")
curr_state=env.reset()
done=False
for _ in tqdm(range(num_collection_steps)):
    action=env.action_space.sample()
    next_state,reward,done,info=env.step(action)
    agent.memory.push(curr_state,action,reward,next_state,not done)
    curr_state=env.reset() if done==True else next_state


episode=0
episode_reward=0
max_reward=-np.inf
print("Training starts .......")
curr_state=env.reset()
done=False
for step in range(1,num_training_steps+1):
    epsilon=agent.get_epsilon(step,steady_epsilon,exploration_steps)
    action=agent.getAction(curr_state,epsilon)
    next_state,reward,done,info=env.step(action)
    agent.memory.push(curr_state,action,reward,next_state,not done)
    curr_state=next_state
    episode_reward+=reward

    if done:
        episode+=1
        print('on episode {} reward {:.2f}\n'.format(episode,episode_reward))
        logger.log(step,{"episode_reward":episode_reward})
        episode_reward=0
        if episode%test_after_episode==0:
            stat=dqn_test(env,num_test_runs,steady_epsilon,agent)
            logger.log(step,stat)
            if stat["avg_reward"]>max_reward:
                max_reward=stat["avg_reward"]
                print(f"saving dqn model in step {step}")
                agent.save()
        curr_state=env.reset()

    if step%fixedQ_update_steps==0:
        agent.updateFixedQ()

    if step%learning_steps==0:
        loss=agent.learn(batch_size=batch_size)
        logger.log(step,{"Loss":loss})