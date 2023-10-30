# seed
random_state=101

#Agent
H=[100,100]

#DQN
experiment="DQN" #DQN
memory_size=300000
gamma=0.99
num_collection_steps=50000
num_training_steps=1000000
steady_epsilon=0.01
exploration_steps=300000
fixedQ_update_steps=10000
learning_steps=4
batch_size=256

#Testing DQN
test_after_episode=50
num_test_runs=10

#PPO
num_of_env=4
n_steps=512
epochs=10
steps_per_epoch=4
shuffle_buffer_size=1024
gamma=0.99
lam=0.95
vf_coef=0.5
ent_coef=0.01
learning_rate=0.00025
max_grad_norm=0.5
cliprange=0.2
cliprange_vf=None

#Training PPO
train_for_step=10000000 #10M
#Testing PPO
test_at_iter=5
num_of_test=10