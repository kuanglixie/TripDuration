# Define your goals

This can determine what competition you want to join and what you can get out of your competition.

1. To learn more about an interesting

    - new problems

2. To get acquainted with new software tools
3. To hunt for a medal

    - whether there is some inconsistency between leaderboard and training data

# Working with ideas

1. Organize ideas in some structure

    - with priority

2. Select the most important and promising ideas
3. Try to understand the reasons why something does/doesn't work


# After entering a competition: Everything is a hyperparameter

From the number of features to model configuration

## Sort all parameters by these principles

1. Importance

    - Tune from important to not useful at all
    - Depend on data structure, target, metric

2. Feasibility

    - easy to tune to tunning this can take forever

3. Understanding

    - I know it's doing to I have no idea


# Data loading

1. Do basic preprocessing and convert csv/txt files into hdf5/npy for much faster loading

    - hdf5 for Pandas dataframes
    - npy for numpy array
    
2. Downcast the 64-bit arrays to 32-bits

3. Large datasets can be processed in chunks

# Performance evaluation

1. Extensive validation is not always needed
2. Start with fastest models - LightGBM

    - Switch to tuning the models and sampling, and stacking, only when I am satisfied with feature engineering


# Fast and dirty always better

1. Don't pay too much attention to code quality
2. Keep things simple: save only important things
3. If you feel uncomfortable with given computational resources - rent a larger server
    
# Initial pipeline

1. Start with simple solution
2. Debug full pipeline
3. "From simple to complex"

# Best practices from software development

1. Use good variable name
2. Keep your research reproducibile

    - Fix random seed
    - Write down exactly how many featuers were generated
    - Use version control

3. Reuse code

    - Use same code for train and test
    
# Read papers

1. About metrics optimization and background knowledge

# One pipeline

1. Read forums and examine kernels
2. Start with EDA and a baseline

    - Check if validation is correlated with leaderboard score
    
3. Add featuers in bulks
4. Overfit training set and then try to constraint the model
5. One notebook per submission
6. Before creating a submission restart the kernel

## Code organization: test/val

### Use the same name `train_path` and `test_path` but different values for test/val

1. `test_path = 'data/val/val.csv'`
2. `test_path = 'data/test.csv'`

### Use macros for a frequent code

### Use a library with frequent operations implemented


# KazAnova's competition pipeline

3,4,5: after trying the problem individually (shut from the outside world) for a week or so then kernels are explored too.

Always gain some advantages.

1. Understand the problem 1d

    - type of problem
    - how BIG is the data
    - hardware needed
    - what the metric being test on

2. Exploratory analysis 1-2d

    - plot histograms of variables and whether difference between and test
    - features versus the target variable and vs time
    - univariate predictability metrics
    - binning numerical features and correlation matrices
    
3. Define cv strategy

    - consistensy
    - Is time important
    - Different entities than train
    - Is it completely random

4. Feature engineering (until last 3-4 days)

    - every problem has its own way
    - can be automated

5. modelling (until last 3-4 days)
6. ensembling last 3-4d

    - small data requires simpler ensemble techniques (like averaging)
    - Helps to average a few low-correlated predictions with good scores
    - Stacking process repeats the modeling process
    
7. Tips on collaboration
8. Selection final submissions

    - best submissions locally and best on leaderboard
    - monitor correlations: if correlations are too high and submissions exist with high score but significantly lower correlations, they could be considered too.
    
Add macro on jupyter notebook:

1. `%macro -q __imp 1`
2. `%store __imp`: to remove this line, add the following code to ipython_config.py
3. `!echo "c = get_config()\nc.StoreMagics.autorestore = True" > ~/.ipython/profile_default/ipython_config.py`
