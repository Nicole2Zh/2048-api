import numpy as np
import torch
from .Model import Net
from .RC_Model import RCNN

class Agent:
    '''Agent Base.'''
    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        state_all=[]
        direction_all=[]
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            if self.game.score < 4096:
                direction_all = np.append(direction_all,direction)
                state_all = np.append(state_all,self.game.board)
            else:
                self.game.end=True
            self.game.move(direction)
            n_iter += 1
            if verbose:
                if self.display is not None:
                    self.display.display(self.game)
        return self.game.score, state_all,direction_all

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):
    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
    
class Cnn_Agent(Agent):
    def __init__(self,game,display = None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        model = Net()
        model.eval()
        model.load_state_dict(torch.load("game2048/model_cnn/epoch_90.pkl"))
        self.model = model
    def step(self):
            board = self.game.board
            arr = np.log2(board + (board == 0))
            arr = np.expand_dims(arr, axis=0)
            arr = torch.tensor(arr)
            arr = arr.unsqueeze(dim=0)
            arr = torch.tensor(arr, dtype=torch.float32)
            pred_y = self.model(arr)
            print(pred_y.shape)
            print(pred_y)
            direction = torch.max(pred_y, 1)[1]
            return int(direction)

class RC_Agent(Agent):
    def __init__(self,game,display = None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        model = RCNN()
        model.eval()
        model.load_state_dict(torch.load("game2048/model_rc256/epoch_5.pkl"))
        self.model = model
    def step(self):
            board = self.game.board
            arr = np.log2(board + (board == 0))
            arr = np.expand_dims(arr, axis=0)
            arr = torch.tensor(arr)
            arr = torch.tensor(arr, dtype=torch.float32)
            pred_y = self.model(arr)
            direction = torch.max(pred_y, 1)[1]
            print('direction:', int(direction))
            return int(direction)


