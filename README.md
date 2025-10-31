# Gradient Flossing for LDLT Lipschitz Networks

## Installation

Using Python 3.11 you can install all the dependencies for the project as,

```bash
pip install -r requirements.txt
```

You need to make sure that you have a GPU and you are running Linux to be able to perform if that does not work for you change it so that it is `reduced-overhead` instead. This will help speed up the training.

## Training

### Single training

If you wish to test the training for a single setup then you would run the appropriate setup in the

```bash
cd src
python linear_train.py
```

### Grid search

To generate the grid search outputs that are present in the paper you can run,

```bash
cd src
python grid_search.py
```

which will generate the `runs` output. This currently does not have an argument parse for improved testing setup so you would have to edit the parameters directly from the command line.