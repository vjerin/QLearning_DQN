import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)

import cv2
from PIL import Image
from matplotlib import style


style.use("ggplot")

#print("done")

os.system('clear')

SIZE = 10

HM_EPISODES = 300
MOVE_PENALTY = 1  # feel free to tinker with these!
ENEMY_PENALTY = 300  # feel free to tinker with these!
FOOD_REWARD = 25  # feel free to tinker with these!
epsilon = 0  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1  # how often to play through env visually.

start_q_table = 'qtable-1.pickle'

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict! Using just for colors
d = {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red



class Blob:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)


    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    
    def action(self,choice):

        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)


    def move(self, x=False,y=False):

        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

'''
# testing 

player = Blob()
food = Blob()
enemy = Blob()


print(player)
print(food)
print(enemy)
print(player-food)
print(player-enemy)
player.move()
print("moving-----------")
print(player)
print(player-food)
print(player-enemy)
player.action(2)
print("moving------------")
print(player-food)
print(player-enemy)

'''

if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]


else:
    with open(start_q_table,"rb") as f:
        q_table = pickle.load(f)


#print(q_table[((0,0),(1,2))])


episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0 and episode!=0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True

    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food,player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        
        else:
            action = np.random.randint(0,4)

        player.action(action)

        
        #### MAYBE ###
        enemy.move()
        food.move()
        ##############

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food, player-enemy)  # new observation
        max_future_q = np.max(q_table[new_obs])  # max Q value for this new obs
        current_q = q_table[obs][action]  # current Q for our chosen action

        if reward == FOOD_REWARD:
           new_q = FOOD_REWARD
        
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY

        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        

        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    
        episode_reward += reward
        if  reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *=EPS_DECAY


moving_avg = np.convolve(episode_rewards,np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f'reward {SHOW_EVERY} MA')
plt.xlabel("episodes")
plt.show()

with open(f"qtable-with-smartFE.pickle","wb") as f:
    pickle.dump(q_table,f)


                    
        
    


