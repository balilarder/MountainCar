import gym
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

class Qlearning(object):
    position_state = 10
    velocity_state = 10

    EPSILON = 0.7              # how greedy
    GAMMA = 0.99
    LEARNING_RATE = 0.01

    train_episode = 5000
    step_limit = 2000

    action_space = [0, 1, 2]
    max_position = 0.6
    min_position = -1.2
    goal_position = 0.5
    max_velocity = 0.07

    def bulid_table(self):
        """
        give position bound and velocity bound to segment the state
        """
        self.qtable = np.zeros((self.position_state, self.velocity_state, 3))
        # 2d table segment
        self.d1 = np.linspace(
            self.min_position, self.max_position, self.position_state+1)
        self.d2 = np.linspace(-self.max_velocity,
                              self.max_velocity, self.velocity_state+1)

        return self.qtable

    def to_state(self, position, velocity):
        p_state = 0
        v_state = 0
        for i, j in enumerate(self.d1):
            # print(i, j)
            if j >= position:
                p_state = i-1
                break
        for i, j in enumerate(self.d2):
            # print(i, j)
            if j >= velocity:
                v_state = i-1
                break
        return (p_state, v_state)

    def choose_action(self, p_state, v_state, qtable, i_episode):
        # lookup self.table and choose an action
        # 2 case: greedy or random

        EPSILON_ = (i_episode*(1-self.EPSILON) / self.step_limit) + self.EPSILON

        state_actions = qtable[p_state][v_state]
        
        if (np.random.uniform() > EPSILON_) or (all(a == 0 for a in state_actions)):
            action = np.random.choice(self.action_space)
            # print("random", action, action)
        else:
            action = max(state_actions)
            action = [i for i, j in enumerate(state_actions) if j == action][0]
            # print("greedy", action)
        return action

    def update_qtable(self, qtable, cur_p, cur_v, new_p, new_v, this_a, exp_a, reward):
        # qtable[3][7][1] += 1
        qtable[cur_p][cur_v][this_a] = qtable[cur_p][cur_v][this_a] + self.LEARNING_RATE*(reward + self.GAMMA*qtable[new_p][new_v][exp_a] - qtable[cur_p][cur_v][this_a])
        # print(qtable)
        # exit()


model = Qlearning()
qtable = model.bulid_table()
# print(qtable)
# qtable[0][2][2] = 3
# print(qtable)
# print(model.position_state)


# Training:
print(model.d1)
print(model.d2)

total_reward_list = []
total_step_list = []

for i in range(model.train_episode):
    total_reward = 0
    total_step = 0

    observation = env.reset()

    # print(observation)
    for j in range(model.step_limit):
        # env.render()
        # print(i, j)
        cur_position = observation[0]
        cur_velocity = observation[1]
        cur_p, cur_v = model.to_state(cur_position, cur_velocity)


        # choose an action, and do
        a = model.choose_action(cur_p, cur_v, qtable, i)
        # print(cur_position, cur_velocity, cur_p, cur_v, a)

        observation_after, reward, done, info = env.step(a)
        total_step += 1
        new_position = observation_after[0]
        new_velocity = observation_after[1]
        new_p, new_v = model.to_state(new_position, new_velocity)

        # print(observation_after, reward, done, info, i, j)
        
        # # setting extra reward:
        # reward -= 3*(new_position-0.5)        # close to goal is better
        
        # if new_position != -0.5:              # no stop in lowest
        #     reward + 10*(abs(new_position+0.5))
        reward += (0.1/(abs(new_position-0.5)) -math.cos(new_position+0.5))
        
        # if new_position >= 0.5:
        #     reward += 10

        total_reward += reward
        # use new state to choose a expect action:
        a_expect = model.choose_action(new_p, new_v, qtable, i)

        # update qtable
        # print("update qtable")
        model.update_qtable(qtable, cur_p, cur_v, new_p, new_v, a, a_expect, reward)

        observation = observation_after
        
        # if finish then re-train gogogo
        if new_position >= 0.5:
            total_step_list.append(total_step)
            total_reward_list.append(total_reward)
            print(i, "finish in", total_reward, total_step)
            break

# print(total_reward_list)
# plt.subplot(2,1,1)
# plt.plot(total_reward_list)
# plt.subplot(2,1,2)
# plt.plot(total_step_list)
# plt.show()

# moving average:
N = 5
total_reward_list = np.convolve(total_reward_list, np.ones((N,))/N, mode='valid')
total_step_list = np.convolve(total_step_list, np.ones((N,))/N, mode='valid')

fig = plt.figure()
x = range(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(total_reward_list)
ax2.plot(total_step_list)

ax1.set_xlabel('episode')
ax1.set_ylim(top=1000)
ax2.set_xlabel('episode')

ax1.set_ylabel('reward')
ax2.set_ylabel('#step')

plt.show()
