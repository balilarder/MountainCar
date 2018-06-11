import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')


# 2D env observation: V=(-0.07, 0.07), P=(-1.2, 0.6)
# Action=[0, 1, 2], respectively to decelerate, no move, accelerate.
# reward = -1 for each time step, so it will be better for earlier finish

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         # env.render()
#         print(observation)
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break


# Q-learning
actions = [0, 1, 2]
state_number = 10            # position segment
EPSILON = 0.9                # how greedy
GAMMA = 0.99
LEARNING_RATE = 0.2

def build_q_table(n_states, n_actions):
    # table = pd.DataFrame(
    #     np.zeros((n_states, len(n_actions))),
    #     columns=n_actions
    # )
    table = {}
    for i in range(n_states):
        table[i] = [0, 0, 0]
    return table

def choose_action(state_number, Qtable):
    state_action = Qtable[state_number]
    if (np.random.uniform() > EPSILON) or (all(a == 0 for a in state_action)):
        action = np.random.choice(actions)
        # print("random", action)
    else:
        action = max(state_action)
        action = [i for i, j in enumerate(state_action) if j == action][0]
        # print("greedy", action)
        
    return action

def position_state(current_position, state_number):
    # a = (current_position+1.2)/1.8
    for i in range(state_number):
        if current_position <= -1.2 + ((i+1)*(1.8/state_number)):    
            return i

print("Q learning")
Qtable = build_q_table(state_number, actions)
print(Qtable)

# Qtable[0][2] = 1
# print(Qtable)
# exit()

observation = env.reset()
print(observation)
# env.render()
# training: fill the table
total_reward_list = []
total_step_list = []

for i_episode in range(200):
    total_reward = 0
    
    observation = env.reset()

    for t in range(5000):
        # env.render()
        position = observation[0]
        velocity = observation[1]
        cur_state = (position_state(position, state_number))
        action = choose_action(cur_state, Qtable)        # choose action
        
        # print("current S: ", cur_state)
        # print("choose A:  ", action)
        # do action
        observation, reward, done, info = env.step(action)
        new_state = position_state(observation[0], state_number)

        # # RIGHT, WANT TO LEFT
        # if position_state(new_state, state_number) == 4 and action == 0:
        #     reward += 10
        # elif position_state(new_state, state_number) == 5 and action == 0:
        #     reward += 20
        # elif position_state(new_state, state_number) == 6 and action == 0:
        #     reward += 30
        # elif position_state(new_state, state_number) == 7 and action == 0:
        #     reward += 40

        # # LEFT, WANT TO RIGHT
        # if position_state(new_state, state_number) == 3 and action == 2:
        #     reward += 10
        # elif position_state(new_state, state_number) == 2 and action == 2:
        #     reward += 20
        # elif position_state(new_state, state_number) == 1 and action == 2:
        #     reward += 30
        # elif position_state(new_state, state_number) == 0 and action == 2:
        #     reward += 40

        # if action == 1:
        #     reward -= 1
        distance = observation[0]+0.5
        if distance > 0:
            if action == 0:
                reward += 3*distance
        else:
            if action == 2:
                reward -= 3*distance
        
        distance_goal = observation[0] - 0.6
        if distance_goal != 0:
            reward += cur_state*(1.0/distance_goal)

        # print(reward)
        # get velocity
        new_velocity = observation[1]
        # print(velocity, new_velocity)
        # print(t, t, t, i_episode)
        # find (s', a')

        if new_state not in Qtable:
            print("start new episode", i_episode)
            total_reward_list.append(total_reward)
            break       # new episode
        a_ = max(Qtable[new_state])
        new_action = [i for i, j in enumerate(Qtable[new_state]) if j == a_][0]
        # print(new_action)

        if new_velocity > 0 and new_action == 2:
            reward += 10
        elif new_velocity > 0 and new_action == 0:
            reward -= 10
        elif new_velocity < 0 and new_action == 0:
            reward += 10
        elif new_velocity < 0 and new_action == 2:
            reward -= 10

        total_reward += reward

        # print(cur_state, action)
        # print(Qtable)
        # print(Qtable[cur_state][action])
        # print(Qtable[new_state][new_action])
        # exit()
        Qtable[cur_state][action] = Qtable[cur_state][action] + LEARNING_RATE*(reward + GAMMA*Qtable[new_state][new_action] - Qtable[cur_state][action])
        
        # print(observation, reward, done)

print(total_reward_list)
plt.plot(total_reward_list)
plt.show()