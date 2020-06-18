import gym
import os
import numpy  as np
import matplotlib.pyplot as plt

os.system('clear')


env =gym.make('MountainCar-v0')
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 26000 # will never render
STATS_EVERY = 500

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}


'''
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

'''

DIS_OS_size  = [20] * len(env.observation_space.high)
dis_os_window_size = (env.observation_space.high - env.observation_space.low)/DIS_OS_size
#print(dis_os_window_size)


epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low = -2, high = 0, size = DIS_OS_size + [env.action_space.n])
#print(q_table.shape)
#print(q_table)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / dis_os_window_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward = 0

    if episode % SHOW_EVERY == 0 and episode != 0:
        #print(episode,'  epsilon=',epsilon)
        render = True
    else:
        render = False    
    discrete_state = get_discrete_state(env.reset())
    #print(env.reset())
    #print(discrete_state)
    #print(q_table[)discrete_state])
    #print(np.argmax(q_table[discrete_state]))

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)    
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        #print(new_state,reward)
        new_discrete_state = get_discrete_state(new_state)
        if render == True:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            #print(f"We made it at episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state  

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)
    if episode % STATS_EVERY == 0:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')


env.close()    


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()