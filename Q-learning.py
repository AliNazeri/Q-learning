import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt

inputt = pd.read_csv("SARSA/maps/1.txt", header=None)
inputt = inputt.values.tolist()

grid = []
for i in inputt:
    grid.append(i[0])

res = []
for i in grid:
    tmp = []
    for j in range(len(i)):
        tmp.append(i[j])
    res.append(tmp)
grid = res.copy()

save_grid = copy.deepcopy(grid)

width = len(grid)
height = len(grid[0])

diamonds = []
# optimal for first map is gama = 0.999999 and alpha 0.2
# effect of new values in old ones
alpha = 0.9

# effect of new values
gama = 0.9
# action selection random treshould
trshld = 100
epoch = 15000
number_of_init_rounds = 100
current_location = (0 ,0)
current_diamond = 0
# learning function
q_table = {}

actions = {"up":0, "down":0, "right":0, "left":0, "upleft":0, "upright":0, "downleft":0, "downright":0 , "noap":0}
actions_moves = {"up":(-1 ,0), "down":(1 ,0), "right":(0 ,1), "left":(0 ,-1), "upleft":(-1 ,-1), "upright":(-1 ,1), "downleft":(1 ,-1), "downright":(1 ,1)} #, "noap":(0 ,0)}

element_reward = {'E':0, '*':-25, 'G':0, 'R':0, 'Y':0, 'g':0, 'r':0, 'y':0, 'T':200}

diamond_score = {0: [50, 0, 0, 0],
                 1: [50, 200, 100, 0], 
                 2: [100, 50, 200, 100], 
                 3: [50, 100, 50, 200], 
                 4: [250, 50, 100, 50]
                 }

for i in range(width):
    for j in range(height):
            q_table[(i,j)] = {0:{"dummy":0},
                              1:{"dummy":0},
                              2:{"dummy":0},
                              3:{"dummy":0},
                              4:{"dummy":0}}

def set_keydoor_values():
    for w in ['g','r','y']:
        m_score = 0
        for i in range(width):
            for j in range(height):
                if grid[i][j] != w:
                    continue
                score = 0
                for n in range(-1,2):
                    if i+n >= 0 and i+n <= width-1 :
                        if grid[i+n][j] == 'W':
                            score += 100
                for m in range(-1,2):
                    if i+m >= 0 and i+m <= height-1 :
                        if grid[i][j+m] == 'W':
                            score += 100
                if m_score < score:
                    m_score = score
        element_reward[w] = m_score

def probs(state):
    probs = []
    act = []
    max_seen_val = -1
    
    for i in actions_moves:
        x, y = state[0] + actions_moves[i][0], state[1] + actions_moves[i][1]
        if x < width and x >= 0 and y < height and y >= 0 and grid[x][y] != 'W':
            probs.append(1/seen_state[x][y])
            act.append(i)
            if seen_state[x][y] > max_seen_val:
                max_seen_val = seen_state[x][y]
    
    max_seen_val *= 100
    for i in range(len(probs)):
        probs[i] *= max_seen_val
    
    return act, probs

def do_state(state):
    if grid[state[0]][state[1]] == 'T':
        action = random.choice(list(actions_moves.keys()))
    else:
        action = max(q_table[state][current_diamond], key=q_table[state][current_diamond].get)
        arr = []
        arr.append(action)
        for i in q_table[state][current_diamond]:
            if q_table[state][current_diamond][action] == q_table[state][current_diamond][i]:
                arr.append(i)
        
        action = random.choice(arr)

        if action == "dummy":
            action = random.choice(list(actions_moves.keys()))
    return action

def select_action(state, multiplier):
    if random.uniform(0, 100) < trshld * multiplier:
        a, p = probs(state)
        action = random.choices(a, weights=p)[0]
    else:
        action = max(q_table[state][current_diamond], key=q_table[state][current_diamond].get)
        arr = []
        arr.append(action)
        for i in q_table[state][current_diamond]:
            if q_table[state][current_diamond][action] == q_table[state][current_diamond][i]:
                arr.append(i)
        
        action = random.choice(arr)

        if action == "dummy":
            action = random.choice(list(actions_moves.keys()))
        
    return action

def get_reward(state, next_state, action, score, reward):
    x = abs(state[0] - next_state[0])
    y = abs(state[1] - next_state[1])
    
    ns_x = next_state[0]
    ns_y = next_state[1]

    a_x = state[0] + actions_moves[action][0]
    a_y = state[1] + actions_moves[action][1]

    reward = -1
    if x+y <= 2:
        reward -= x + y
        score -= x + y
    
    if action != "noap":
        if a_x < 0 or a_y < 0 or a_y >= height or a_x >= width or grid[a_x][a_y] == 'W':
            reward -= 10000
    else:
        reward -= 100

    if grid[ns_x][ns_y] in element_reward.keys():
        reward += element_reward[grid[ns_x][ns_y]]
    elif grid[ns_x][ns_y] in ['1','2','3','4']:
        tmp = diamond_score[current_diamond][int(grid[ns_x][ns_y]) - 1]
        score += tmp
        reward += tmp*10

    return reward , score

def remove_obj(x, y, current_diamond, multiplier):
    if grid[x][y] in ['1','2','3','4']:
        tmp = grid[x][y]
        grid[x][y] = 'E'
        multiplier = calculate_epsilon()
        return int(tmp), multiplier
    elif grid[x][y] in ['g','r','y']:
        element_reward[grid[x][y].upper()] = element_reward[grid[x][y]]*10
        grid[x][y] = 'E'
    return current_diamond, multiplier

def do_action(loc, action):
    x,y = loc[0] + actions_moves[action][0] ,loc[1] + actions_moves[action][1]
    if x >= width or x < 0 or y >= height or y < 0 or grid[x][y] == 'W':
        return loc
    elif grid[x][y] in ["G","R","Y"]:
        if keys[grid[x][y].lower()] == 0:
            return loc
        else:
            grid[x][y] = 'E'
    elif grid[x][y] in ["g","r","y"]:
        keys[grid[x][y]] = 1
    return (x,y)
    
def q_update(S, A, R, SS):
    tmp_k = q_table[S][current_diamond].keys()
    if A not in tmp_k:
        q_table[S][current_diamond][A] = 0
        if "dummy" in tmp_k:
            del q_table[S][current_diamond]['dummy']
    if grid[S[0]][S[1]] == 'T':
        return
    q_table[S][current_diamond][A] = round(q_table[S][current_diamond][A] + alpha * (R + gama * np.max(list(q_table[SS][current_diamond].values())) - q_table[S][current_diamond][A]), 4)

def finished():
    for i in range(width):
        for j in range(height):
            if grid[i][j] in ['1','2','3','4']:
                return False
    return True

################################################################333
def count_E(x):
    w = 0
    for i in range(x):
        for j in range(height):
            if grid[i][j] == 'W':
                w += 1
    return x*height - w

def calculate_alpha():
    for i in range(width-1,-1,-1):
        for j in range(height-1,-1,-1):
            if grid[i][j] in ['1','2','3','4']:
                return (count_E(i) + 14) / ( width * height )

def calculate_epsilon():
    mn_distance = np.Infinity
    save = 0
    for i in diamonds:
        distance = abs(i[0] - current_location[0]) + abs(i[1] - current_location[1])
        if mn_distance > distance:
            mn_distance = distance
            save = i

    diamonds.remove(save)
    if 30 < mn_distance <= 40:
        return 11/8
    elif 20 < mn_distance <= 30:
        return 10/8
    elif 15 < mn_distance <= 20:
        return  9/8
    elif 10 < mn_distance <= 15:
        return 1
    else:
        return 3/4

################################################################################

seen_state = []
for i in range(width):
    seen_state.append([])
for i in range(width):
    for j in range(height):
        seen_state[i].append(1)

set_keydoor_values()
holes = []
for i in range(width):
    for j in range(height):
        if grid[i][j] in ['1', '2', '3', '4']:
            diamonds.append((i, j))
        if grid[i][j] == 'O':
            holes.append((i, j))

holes.append((0, 0))

sv_diamonds = copy.deepcopy(diamonds)

score_list = []

each_hole_round = round(epoch/len(holes))
sub_trshd = round((trshld*80/100)/epoch, 6)
multiplier = calculate_epsilon()
alpha = round(calculate_alpha(), 4)

max_epoch = 300
counter = 0


sv_score = 0
for hole in holes:
    trshld = 100
    for e in range(each_hole_round):
        diamonds = copy.deepcopy(sv_diamonds)
        grid = copy.deepcopy(save_grid)
        current_location = hole
        reward = 0
        score = 0
        current_diamond = 0
        keys = {'g':0,'r':0,'y':0}

        for i in range(number_of_init_rounds):
            if finished():
                break
            
            action = select_action(current_location, multiplier)

            location = do_action(current_location, action)        
        
            reward, score = get_reward(current_location, location, action, score, reward)

            next_diamond, multiplier = remove_obj(location[0], location[1], current_diamond, multiplier)

            q_update(current_location, action, reward, location)
        
            current_diamond = next_diamond
            current_location = location
            seen_state[location[0]][location[1]] += 1

        
        if sv_score <= score:
            sv_score = score
            counter += 1
            if counter == max_epoch:
                break
        trshld -= sub_trshd

        score_list.append(score)
plt.plot(score_list)

plt.show()
#############################################################
# Second turn do the playing

def remove_obj2(x, y):
    if grid[x][y] in ['1','2','3','4',1,2,3,4]:
        tmp = grid[x][y]
        grid[x][y] = 'E'
        return int(tmp),True
    elif grid[x][y] in ['g','r','y']:
        grid[x][y] = 'E'
    return current_diamond, False

grid = copy.deepcopy(save_grid)
current_location = (0 ,0)
current_diamond = 0
keys = {'g':0,'r':0,'y':0}
score = 0
trshld = 0

print(current_location)
for i in range(number_of_init_rounds):
    print("a : ",action)
    print(q_table[current_location][current_diamond])
    # q_table[current_location][current_diamond][action] -= 50
    print(keys)
    action = select_action(current_location, multiplier)
    q_table[current_location][current_diamond][action] -= 50
    location = do_action(current_location, action)
    
    reward, score = get_reward(current_location, location, action, score, reward)

    pre_d = current_diamond

    next_diamond, t = remove_obj2(location[0], location[1])

    if(t):
        score += diamond_score[pre_d][current_diamond-1]

    q_update(current_location, action, reward, location)

    current_diamond = next_diamond
    current_location = location

    print(current_location,"d : ",current_diamond)
    for i in grid:
        print(i)
    input()

    if finished():
        break

#         score_list.append(score)
# plt.plot(score_list)

# plt.show()