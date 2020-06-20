# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
    * [`data_gene.py`](game2048/data_gene.py): code to generate data from ExpectiMaxAgent.
    * [`RC_Model.py`](game2048/RC_Model.py): model that merge RNN and CNN.
    * [`train_RC.py`](game2048/train_RC.py): code to train RC Model.
    * [`CNN_Model.py`](game2048/CNN_Model.py): CNN Model.
    * [`train_CNN.py`](game2048/train_CNN.py): code to train CNN Model.
    * [`model_rc256`](game2048/model_rc256): result model that I train.
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.

#To generate data
```
python data_gene.py
```

#To train model
```
python train_RC.py
```

#To evaluate the model
```
python evaluate.py >> EE369_evaluation.log
```

#To generate fingerprint
```
python generate_fingerprint.py
```

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 / EE228 students from SJTU
Please read course project [requirements](EE369.md) and [description](https://docs.qq.com/slide/DS05hVGVFY1BuRVp5). 
