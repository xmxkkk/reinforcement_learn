import numpy as np
import pandas as pd
import time

# print(np.array([[1,1,1],[0,0,0]])[0].shape[0])
# exit()


EPSILON=0.9
ACTIONS=['left','right','up','down']


FRESH_TIME=0.1
MAX_EPISODES=2000000
ALPHA=0.1
LAMBDA=0.9


# MAP=np.array([
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
#         [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
#         [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
#         [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#         [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
#         [0, 0, 1, 0, 0, 1, 0, 0, 0, 2],
#         [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
#         [0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])

# MAP=np.array([
#         [0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 1, 0, 1, 0, 1, 1, 0],
#         [0, 0, 0, 1, 0, 1, 1, 0],
#         [1, 0, 0, 0, 0, 1, 1, 0],
#         [0, 0, 1, 1, 1, 0, 1, 0],
#         [0, 1, 0, 0, 0, 0, 1, 0],
#         [0, 1, 0, 1, 0, 1, 0, 1],
#         [0, 0, 0, 1, 0, 0, 0, 2],
#     ])
# MAP=np.array([
#         [0, 0, 1, 0, 0, 0,],
#         [0, 1, 0, 0, 1, 0],
#         [0, 0, 1, 0, 1, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 1, 0, 1, 2, 0],
#         [0, 0, 0, 1, 0, 0],
#     ])
# MAP=np.array([
#         [0, 0, 0, 0],
#         [0, 0, 1, 0],
#         [0, 1, 2, 1],
#         [0, 0, 0, 0],
#     ])

MAP=np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 2],
    ])

WIDTH=MAP.shape[0]
HEIGHT=MAP.shape[1]


def build_q_table():
    return np.zeros((WIDTH,HEIGHT,4))

def choose_action(x,y,q_table,test=False):
    state_actions=q_table[y][x]

    while True:
        if ((np.random.uniform()>EPSILON) or (state_actions.sum()==0)) and not test:
            action_name=np.random.choice(ACTIONS)
        else:
            val=state_actions[state_actions.argmax()]
            lst=[]
            for v in range(len(state_actions)):
                if state_actions[v]==val:
                    lst.append(v)
            if len(lst)==1:
                action_name=ACTIONS[state_actions.argmax()]
            else:
                idx=np.random.choice(lst)
                action_name=ACTIONS[idx]

        if x==0 and action_name=='left':
            continue
        if x==WIDTH-1 and action_name=='right':
            continue
        if y==0 and action_name=='up':
            continue
        if y==HEIGHT-1 and action_name=='down':
            continue

        break

    return action_name

TERMINAL=False

def get_env_feedback(x,y,A):
    R=0
    if MAP[y][x]==1:
        R=-0.5
    elif MAP[y][x]==2:
        R=1

    if A=='right':
        x+=1
    elif A=='left':
        x-=1
    elif A=='up':
        y-=1
    elif A=='down':
        y+=1

    return x,y,R

def update_env(x,y,episode,step_counter):
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if i==y and j==x:
                print('P',end=',')
            else:
                print(MAP[i][j],end=',')
        print()
    print('=============================')

def printQ(q_table):
    print('+++++++++++++++++++++++++++++++++++++++++')
    for i in range(HEIGHT):
        for j in range(WIDTH):
            print(q_table[i][j],end='|')
        print()

def rl():
    q_table=build_q_table()
    for episode in range(MAX_EPISODES):
        step_counter=0
        x=0
        y=0
        is_terminated=False
        # update_env(x,y,episode,step_counter)
        while not is_terminated:
            if episode==MAX_EPISODES-1:
                A = choose_action(x,y,q_table,test=True)
            else:
                A = choose_action(x, y, q_table, test=False)
            x_,y_,R=get_env_feedback(x,y,A)

            idx=ACTIONS.index(A)
            q_predit=q_table[y][x][idx]

            if R==1 or R<0:
                q_target = R
                is_terminated = True
            else:
                q_target = R + LAMBDA * q_table[y_][x_].max()

            q_table[y][x][idx]+=ALPHA*(q_target-q_predit)

            x=x_
            y=y_

            # update_env(x,y,episode,step_counter+1)
            step_counter+=1

            if R==1:
                print('episode={},step={}'.format(episode,step_counter))

        if episode%1000==999:
            printQ(q_table)
    return q_table
            # print(step_counter)

'''
def rl2():
    q_table=build_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter=0
        S=0
        is_terminated=False
        update_env(S,episode,step_counter)
        while not is_terminated:
            A=choose_action(S,q_table)
            S_,R=get_env_feedback(S,A)
            q_predit=q_table.ix[S,A]
            if S_!='terminal':
                q_target=R+LAMBDA*q_table.iloc[S_,:].max()
            else:
                q_target=R
                is_terminated=True

            q_table.ix[S,A]+=ALPHA*(q_target-q_predit)
            S=S_

            update_env(S,episode,step_counter+1)
            step_counter+=1
    return q_table'''
q_table=rl()
# print(q_table)

