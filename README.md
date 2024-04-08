# DSDN

This method does link predition task based on the topology of local enclosing subgraph.

There are three versions provided:

- main-fixed.py: making predictions by entering fixed parameters

- main-auto.py & main-auto-full.py: automatically selecting the best combination of parameters for link prediction based on the validation set

## Requirements installation

execute the command below to install required environments:

`pip install -r requirements.txt`

**Do a simple test**:

`python main-fixed.py --data-name NS --c 0.75 --hop 1 --bins-num 900 --h-init 0.025`

`python main-fixed.py --data-name citeseer --c 0.65 --hop 2 --bins-num 900 --h-init 0.025`

## main-auto.py & main-auto-full.py: start link prediction automatically 

Type the command below to have a simple try.

`python main-auto.py --data-name *Power* --use-splitted *True*`

**tips:** parameters marked with * can be changed.

### the procedure of selecting parameters 

**take main-auto.py as an example**

- randomly sample validation set from training set(*5%*).
- fix *c=0.7*, *#bin=900* and *h_init=0.025*, do training on *hop = 1, 2, 3*.
  - pick up the best *hop* by comparing aucs of validation set.
- fix *c=0.7* and the best *hop*, do training on *#bins=400,900* and *h_init=0.05,0.025*.
  - pick up the best *#bins*, *h_init* in the same way.
- now the best *hop* and *(#bins, h_init)* are selected, then do training on different *c* in [0.4,0.45,0.5,...,0.75,0.8].
  - pick up the best *c* in the same way.
- retrain the whole model with those selected parameters.

**tips:** main-auto-full.py does the same thing but with a wider range of parameters.

## main-fixed.py: start link prediction manually 

If you want to select parameters manually, try:

`python main-fixed.py --data-name *NS* --use-splitted *True* --hop *1* --c *0.45* --bins-num *400* --h-init *0.025*`

**tips:** parameters marked with * can be changed.

### Parameters

- --data-name: dataset used for link prediction. *default = NS*
  - NS, Power, Router, GRQ, cora, citeseer, Email...
- --hop: the enclosing subgraph hop number. *default = 2*
- --bins-num: The number of bins evenly distributed in the 2-D plane. *default = 900*
- --h_init: the initialization of Gauss sensors' bandwidth. *default = 0.025*
- --c: 1-c is the restart probability of RWR. *default = 0.7*
  - attention: *c* must be in range(0,1).
- --max-nodes-per-hop: the maximum number of nodes included in each hop. *default = 100*
- --use-splitted: wheather to use data that has been splitted. Data will be splitted randomly when it is False. *default = True*
- --split-index: the index of splitted data when `use-splitted` is True. *default = 1*
- --test-ratio: test ratio of dataset when `use-splitted` is False. *default = 0.1*
- --save-path: the saving path of link prediction results. *default = result/result.txt*
- --if-save: save results or not. *default = True*

