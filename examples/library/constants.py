# seed
random_state=101

#Agent
H=[100,100]

#DQN
experiment="DQN" #DQN
memory_size=100000
gamma=0.99
num_collection_steps=50000
num_training_steps=1000000
steady_epsilon=0.01
exploration_steps=200000
fixedQ_update_steps=10000
learning_steps=4
batch_size=32

#Testing
test_after_episode=50
num_test_runs=10