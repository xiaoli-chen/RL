#!/usr/bin/env python
# coding:utf-8




from __future__ import print_function
import sys
sys.path.append("game/")
import bird_wrap as game
import argparse
import random
import numpy as np
from collections import deque
import pygame
import json
from sys import exit # the exit function is helpful and necessary
from pygame.locals import *


GAME = 'bird' # name in the log file
CONFIG = 'nothreshold'
ACTIONS = 2 # number of effective actions

# this methods is not Q-learning!
# it only distribute the final rewards along the fly-path.
# 

def playGame(args):
    delta_x = 200
    delta_y = 0 
      
    if args['mode'] == 'Run':
        
        print ("Now we load weight from pretrained_model")
        states = np.load('./pretrained_model/bird51.npy')

    else:  # train
        states = 20*np.ones([40,15,2])
        states[:,:,1]=5           
    
    game_state = game.GameState() 
     



    # initialize 
    # i,j,a
    i = int(delta_y//20 + 25)
    j = int(delta_x//20)


    # fly_path
    fly_path = []
    terminal = False
    gamma = 0.95
    count = 0

    while True:

        
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        reward = 0
        
        i = int(delta_y//20 + 25)
        j = int(delta_x//20)
        
        # the decision of direction of fly is determined by environment states. H
        # here we didn't use greedy stratedy. The greedy stratedy will fall into the local minimal.
        # we are making decision according to the relative ratio of the possible choice in the next step
        # as the possibility. It is similar to the greedy stratedy, but will explore more space.
        
        ratio = states[i,j,1]/(states[i,j,0]+states[i,j,1])
        print(ratio)
        if random.random()<ratio:
            action = 1
        else:
            action = 0

        # save the fly_path in a list
        if terminal==False:
            fly_path.append([i,j,action])

        #a_t = random.randint(0,2)
        
        image_data,reward,terminal,[delta_x,delta_y]= game_state.frame_step(action)
        
        # update q(s,a) along the fly_path. It is not Q-learning! It is only distribute the final
        # rewards along the path, and let it accumulate. 
        if terminal==True or reward==1 or reward == -1:
            for iter,index in enumerate(fly_path):
                i_index = index[0]
                j_index = index[1]
                a_index = index[2]
                states[i_index,j_index,a_index] += reward * pow(gamma,(len(fly_path)-iter))
                if states[i_index,j_index,a_index]<0:
                    states[i_index,j_index,a_index]=1
            fly_path=[]
            count +=1

        if count%100==99:
            if args['mode'] == 'Train': 
                name = 'bird%d.npy'%(count//100)
                np.save(name,states)
            
        #game.showScore('5')


def main():
    parser = argparse.ArgumentParser(description='flappy_bird')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()
