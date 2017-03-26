#!/usr/bin/env python
from sys import argv
import numpy as np

LEFT='L'
RIGHT='R'
UP='U'
DOWN='D'

mod=0

GOAL3="1,2,3,4,5,6,7,8,0";
GOAL4="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0";

def move():
    if len(argv) < 3:
        print "Warning: arguments invalid. [tile_file_name] [step_file_name]."
        return;
    script, fname, step_fname = argv

    
    file = open(fname)
    tiles_line = file.readlines()
    matrix=[]
    for line in tiles_line:
        items = line.split('\n')[0].split(',')
        for item in items:
            matrix.append(item)
    
    #print matrix
    tiles_line = [x if (x != '') else '0' for x in matrix]


    tiles = map(int, tiles_line)  # read the original state
    if len(tiles) == 0:
        print "Warning: invalid argument. Should contain 3 lines"
        return

    moves = open(step_fname).readline().split(',')

    global mod
    if len(tiles) == 9:
        mod = 3
        goal = map(int, GOAL3.split(","))    # the supposed final state
    elif len(tiles) == 16:
        mod = 4
        goal = map(int, GOAL4.split(","))    # the supposed final state
    else:
        print "Warning: invalid argument. Should have 9 or 16 elements."
        return

    #print tiles, goal, mod, moves
    start_moving(moves, tiles, goal)


def start_moving(moves, tiles, goal):
    #print "Now start moving ....."

    n_moves = len(moves)
    #print "There are %i steps to move." % n_moves
    index = tiles.index(0)
    #print "Blank tile is located at:", index

    for move in moves:
        if move == LEFT:
            tiles, index = move_left(tiles, index)

        elif move == RIGHT:
            tiles, index = move_right(tiles, index)

        elif move == UP:
            tiles, index = move_up(tiles, index)

        elif move == DOWN:
            tiles, index = move_down(tiles, index)
        #print tiles

    if np.array_equal(tiles, goal):
        print bcolors.OKGREEN + "Passed. Moving Succeed!\n" + bcolors.ENDC
        #print "Passed" 
    else:
        print bcolors.FAIL + "Moving Failed!\n" + bcolors.ENDC
        #print "Failed"

def move_left(array, idx):
    #print "Moving left..."
    if idx % mod == 0:
        print "You've reached the wall, stop moving."
    else:
        array[idx-1], array[idx] = array[idx], array[idx-1]
    idx = idx - 1
    return (array, idx)


def move_right(array, idx):
    #print "Moving right..."
    if (idx + 1) % mod == 0:
        print "You've reached the wall, stop moving."
    else:
        array[idx+1], array[idx] = array[idx], array[idx+1]
        idx = idx + 1
    return (array, idx)

def move_up(array, idx):
    #print "Moving upward..."
    if (idx - mod) < 0:
        print "You've reached the wall, stop moving."
    else:
        array[idx-mod], array[idx] = array[idx], array[idx-mod]
        idx = idx - mod
    return (array, idx)

def move_down(array, idx):
    #print "Moving downward..."
    if (idx + mod) > (mod*mod):
        print "You've reached the wall, stop moving."
    else:
        array[idx+mod], array[idx] = array[idx], array[idx+mod]
        idx = idx + mod
    return (array, idx)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


move()

