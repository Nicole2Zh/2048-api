from game import Game
from displays import Display
import numpy as np
import os
import json

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    score, board, direction = agent.play(verbose=False)
    return game.score, board, direction

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 4096
    N_TESTS = 100

    '''====================
    Use your own agent here.'''
    from agents import ExpectiMaxAgent as TestAgent
    '''===================='''

    scores = []
    all_board =[]
    all_direction =[]
    for j in range(N_TESTS):
        all_board = []
        all_direction = []
        for i in range(50):
            score, state_all, direction_all = single_run(GAME_SIZE, SCORE_TO_WIN,
                               AgentClass=TestAgent)
            scores.append(score)
            board_name = "data/data"+str(j)+".npy"
            direction_name = "data/direction"+str(j)+".npy"
            np.save(board_name,all_board)
            np.save(direction_name,all_direction)
            all_board = np.load(board_name)
            all_board = np.append(all_board, state_all)
            np.save(board_name, all_board)
            all_direction = np.load(direction_name)
            all_direction = np.append(all_direction, direction_all)
            np.save(direction_name, all_direction)

            all_board = np.load(board_name)
            all_direction = np.load(direction_name)
            num = int(np.size(all_board)/16)
            all_board = all_board.reshape(num,4,4)
            all_direction = all_direction.reshape(num,1)
            np.save(board_name, all_board)
            np.save(direction_name, all_direction)