#!/usr/bin/env python
# coding: utf-8

# # Programming Project 3 
# ## 605.649 Introduction to Machine Learning
# ## Ricca Callis

# ## Directions
# 
# The purpose of this assignment is to give you a chance to get some hands-on experience learning decision
# trees for classification and regression. This time around, we are not going to use anything from the module
# on rule induction; however, you might want to examine the rules learned for your trees to see if they “make
# sense.” Specifically, you will be implementing a standard univariate (i.e., axis-parallel) decision tree and willcompare the performance of the trees when grown to completion on trees that use either early stopping (for
# regression trees) or reduced error pruning (for classification trees).
# 
# For decision trees, it should not matter whether you have categorical or numeric attributes, but you need
# to remember to keep track of which is which. In addition, you need to implement that gain-ratio criterion
# for splitting in your classification trees. For the regression trees, all of the attributes will be numeric.
# For this assignment, you will use three classification datasets and three regression data sets that you will
# download from the UCI Machine Learning Repository, namely:
# 
# 1. Abalone — https://archive.ics.uci.edu/ml/datasets/Abalone
# [Classification] Predicting the age of abalone from physical measurements.
# 
# 2. Car Evaluation — https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
# [Classification] The data is on evaluations of car acceptability based on price, comfort, and technical
# specifications.
# 
# 3. Image Segmentation — https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
# [Classification] The instances were drawn randomly from a database of 7 outdoor images. The images
# were hand segmented to create a classification for every pixel.
# 
# 4. Computer Hardware — https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
# [Regression] The estimated relative performance values were estimated by the authors using a linear
# regression method. The gives you a chance to see how well you can replicate the results with these two
# models.
# 
# 5. Forest Fires — https://archive.ics.uci.edu/ml/datasets/Forest+Fires
# [Regression] This is a difficult regression task, where the aim is to predict the burned area of forest
# fires, in the northeast region of Portugal, by using meteorological and other data .
# 
# 6. Wine Quality — https://archive.ics.uci.edu/ml/datasets/Wine+Quality
# [Regression] This contains two data sets, one for red wine and one for white. Either combine the data
# sets into a single set for the regression task or build separate regression trees. This is your choice;
# however, we expect

# ### For this project, the following steps are required:
# 
#  Download the six (6) data sets from the UCI Machine Learning repository. You can find this repository
# at http://archive.ics.uci.edu/ml/. All of the specific URLs are also provided above.
# 
#  Implement the ID3 algorithm for classification decision trees using gain-ratio as the splitting criterion.
# 
#  Implement reduced-error pruning to be used as an option with your implementation of ID3.
# 
#  Run your ID3 implementation on each of the three classification data sets, comparing both pruned
# and unpruned versions of the trees. These runs should be done with 5-fold cross-validation so you can
# compare your results statistically. You should pull out 10% of the data to be used as a validation set
# and then use the remaining 90% for cross validation. You should use classification error for your loss
# function.
# 
#  Implement the CART algorithm for regression decision trees using mean squared error as the splitting
# criterion.
# 
#  Incorporate a cut-off threshold for early stopping. If the threshold is set to zero, this should indicate
# no early stopping.
# 
#  Run your CART implementation on each of the three regression data sets, comparing both full and
# stopped versions of the trees. You will need to tune the stopping threshold and should use the same
# procedure for extracting a validation set to serve as your tuning set. The runs should be done with
# 5-fold cross-validation so you can compare your results statistically. You should use mean squared
# error for your loss function.
# 
#  Write a very brief paper that incorporates the following elements, summarizing the results of your
# experiments.
# 
# 1. Title and author name
# 2. A brief, one paragraph abstract summarizing the results of the experiments
# 3. Problem statement, including hypothesis, projecting how you expect each algorithm to perform
# 4. Brief description of algorithms implemented
# 5. Brief description of your experimental approach
# 6. Presentation of the results of your experiments
# 7. A discussion of the behavior of your algorithms, combined with any conclusions you can draw
# 8. Summary
# 9. References (you should have at least one reference related to each of the algorithms implemented, a reference to the data sources, and any other references you consider to be relevant)
# 
#  Submit your fully documented code, the outputs from running your programs, and your paper. Your
# grade will be broken down as follows:
# 
# – Code structure – 10%
# – Code documentation/commenting – 10%
# – Proper functioning of your code, as illustrated by a 5 minute video – 30%
# – Summary paper – 50%

# In[1]:


# Author: Ricca Callis
# EN 605.649 Introduction to Machine Learning
# Programming Project #3
# Date Created: 7/6/2020
# File name: Programming Assignment 3 - Callis.ipynb
# Python Version: 3.7.5
# Jupyter Notebook: 6.0.1
# Description: Implementation of decision tree classifier and regressor algorithms ID3 & CART 
# using 6 datasets from the UCI Machine Learning Repository

"""
ID3 Algorithm: Algorithm for classification decision trees using gain-ratio as the splitting criterion
"""

"""
CART Algorithm: Algorithm for regression decision trees using mean squared error as the splitting criterion.
"""


"""
Required Data Sets:
    abalone.data.csv
    abalone.names.csv
    car.data.csv
    car.names.csv
    forestfires.data.csv
    forestfires.names.csv
    machine.data.csv
    machine.names.csv
    segmentation.data.csv
    segmentation.names.csv
    winequality-red.csv
    winequality-white.csv
    winequality.names.csv
""" 


# In[2]:


from platform import python_version
print ( python_version() )


# In[3]:


# Common standard libraries
import datetime
import time
import os
# Common external libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import sklearn #scikit-learn
import sklearn
from sklearn.model_selection import train_test_split 
import random as py_random
import numpy.random as np_random
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import scipy.stats as stats
from toolz import pipe # pass info from one process to another (one-way communication)
from typing import Callable, Dict, Union, List
from collections import Counter, OrderedDict
import logging
import multiprocessing
import operator
import sys
import copy
from typing import Callable, Dict, Union
from functools import partial
from itertools import product
import warnings
import io
import requests as r

#logging.basicConfig ( filename ='logfile.txt' )
logging.basicConfig()
logging.root.setLevel ( logging.INFO )
logger = logging.getLogger ( __name__ )

sys.setrecursionlimit ( 10000 )


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Check current directory
currentDirectory = os.getcwd()
print ( currentDirectory )


# In[5]:


# Input data files are available in the ".../input/" directory
# Change the Current working Directory
os.chdir ( '/Users/riccacallis/Desktop/JHU/Data Science/Introduction to Machine Learning/Programming Project 3/input' )

# Get Current working Directory
currentDirectory = os.getcwd()
print ( currentDirectory )


# In[6]:


# List files in input directory
from subprocess import check_output
print ( check_output ( [ "ls", "../input" ] ).decode ( "utf8" ) )


# # Decision Trees
# 
# 
# **Overview:**
# 
# A Decision tree is a non-parametric supervised learning technique. Generally, decision trees employ greedy searches to best partition the feature space so as to explain the target variable.
# 
# The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).
# 
# In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.
# 
# 
# ## Types of Decision Trees
# 
# Types of decision trees are based on the type of target variable we have. It can be of two types:
# 
#     1. Categorical Variable Decision Tree: Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.
#     
#     2. Continuous Variable Decision Tree: Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree.
# 
# ## Terminology
# 
# Root Node: It represents the entire population or sample and this further gets divided into two or more homogeneous sets.
# 
# Splitting: It is a process of dividing a node into two or more sub-nodes.
# 
# Decision Node: When a sub-node splits into further sub-nodes, then it is called the decision node.
# 
# Leaf / Terminal Node: Nodes do not split is called Leaf or Terminal node.
# 
# Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say the opposite process of splitting.
# 
# Branch / Sub-Tree: A subsection of the entire tree is called branch or sub-tree.
# 
# Parent and Child Node: A node, which is divided into sub-nodes is called a parent node of sub-nodes whereas sub-nodes are the child of a parent node.
# 
# Decision trees classify the examples by sorting them down the tree from the root to some leaf/terminal node, with the leaf/terminal node providing the classification of the example. Each node in the tree acts as a test case for some attribute, and each edge descending from the node corresponds to the possible answers to the test case. This process is recursive in nature and is repeated for every subtree rooted at the new node.
# 
# 
# ### ID3
# 
# The Iterative Dichotomizer 3 (ID3) algorithm is a non-parametric decision supervised learning algorithm, originally proposed for classification problems. ID3 builds decision trees using a top-down greedy search approach through the space of possible branches with no backtracking. A greedy algorithm, as the name suggests, always makes the choice that seems to be the best at that moment.
# 
# **Steps in ID3 algorithm:**
# 1. It begins with the original set S as the root node.
# 
# 2. On each iteration of the algorithm, it iterates through the very unused attribute of the set S and calculates Entropy(H) and Information gain(IG) of this attribute.
# 
# 3. It then selects the attribute which has the smallest Entropy or Largest Information gain.
# 
# 4. The set S is then split by the selected attribute to produce a subset of the data.
# 
# 5. The algorithm continues to recur on each subset, considering only attributes never selected before.
# 
# ### CART
# 
# Similar to ID3 but is adapted to regression problems by replacing the entropy expression with MSE.
# 
# ### Attribute Selection Measures
# 
# **Entropy** 
# 
# Entropy is a measure of the randomness in the information being processed. The higher the entropy, the harder it is to draw any conclusions from that information.
# 
# ID3 follows the rule — A branch with an entropy of zero is a leaf node and A brach with entropy more than zero needs further splitting.
# 
# **Gini Index**
# The Gini index is a cost function used to evaluate splits in the dataset. It is calculated by subtracting the sum of the squared probabilities of each class from one. It favors larger partitions and easy to implement whereas information gain favors smaller partitions with distinct values. Gini Index works with the categorical target variable “Success” or “Failure”. It performs only Binary splits.The higher the value of Gini index, the higher the homogeneity.
# 
# Steps to Calculate Gini index for a split:
# 
# 1. Calculate Gini for sub-nodes, using the above formula for success(p) and failure(q) (p²+q²).
# 
# 2. Calculate the Gini index for split using the weighted Gini score of each node of that split.
# 
# CART (Classification and Regression Tree) uses the Gini index method to create split points.
# 
# **Information Gain**
# Information gain or IG is a statistical property that measures how well a given attribute separates the training examples according to their target classification. Constructing a decision tree is all about finding an attribute that returns the highest information gain and the smallest entropy.
# 
# Information gain is a decrease in entropy. It computes the difference between entropy before split and average entropy after split of the dataset based on given attribute values. 
# 
# ID3 decision tree algorithm uses information gain.
# 
# **Gain Ratio**
# Gain ratio overcomes the problem with information gain by taking into account the number of branches that would result before making the split. It corrects information gain by taking the intrinsic information of a split into account.
# 
# ### Avoiding Overfitting: Pruning
# 
# The splitting process results in fully grown trees until the stopping criteria are reached. But, the fully grown tree is likely to overfit the data, leading to poor accuracy on unseen data.
# 
# In pruning, you trim off the branches of the tree, i.e., remove the decision nodes starting from the leaf node such that the overall accuracy is not disturbed. This is done by segregating the actual training set into two sets: training data set, D and validation data set, V. Prepare the decision tree using the segregated training data set, D. Then continue trimming the tree accordingly to optimize the accuracy of the validation data set, V.

# In[7]:


# Building Decision Tree
# Attribute Selection Metrics

# Entropy used for ID3 Algorithm (Decision Tree Classification)
def entropy ( target_array ):
    """
    Function finds entropy of a given set of class labels
        Parameters:
            target_array: Indicates "y", or the target array. (Type: Array; np.ndarray) 
    """
    return -1 * sum (
        [
            pipe ( np.sum ( target_array == value ) / len ( target_array ), lambda ratio: ratio * np.log ( ratio ) )
            for value in set ( target_array )
        ]
    ) # End entropy()


def negative_mse ( target_array ):
    """
    Function finds the negative mean squared distance from mean for point in Y (variance)
        Parameters:
            target_array: Indicates "y", or the target array. (Type: Array; np.ndarray) 
    """
    return -1 * mse ( target_array )
    # End negative_mse()

# Mean Squared Error (mse) used for CART algorithm (Decision Tree Regression)
def mse ( target_array ):
    """
    Function finds mean squared distance from mean for point in Y (variance)
        Parameters:
            target_array: Indicates "y", or the target array. (Type: Array; np.ndarray) 
    """
    return np.mean ( ( target_array - np.mean ( target_array ) ) ** 2 )
    # End mse()

# Building Decision Tree
# Nodes

'''''
Class: TreeSplits
    - This class contains functions for decision tree nodes

Functions:
    -__init__: Initializes the TreeSplits class
    - updateTreeValues: Function to update the values of the TreeSplits object  
    - isNodeLeaf: Function to determine whether a node is a leaf (i.e., has no children)

'''''
class TreeSplits:

    def __init__ (
        # Initialize parameters
        self,
        feature_column = None,
        feature_value = None,
        node_type = None,
        nodes = None,
        children = [],
    ):
        self.feature_column = feature_column  # Column index of split
        self.feature_value = feature_value  # Value of split (for continuous)
        self.node_type = node_type  # Continuous or discrete
        self.nodes = nodes  # Children nodes
        self.children = children  # Target values
        # End __init__()

    def updateTreeValues ( self, feature_column, feature_value, node_type, nodes, children = [ ] ):
        """
        Function to update the values of a TreeSplits object.
            Parameters:
                self: Indicates class instance
                feature_column: Indicates column index of split
                feature_value: Indicates the value of the split (for continuous data)
                node_type: Indicates whether the node is continuous or discrete
                nodes: Indicates children nodes
                children: Indicates target values
            Returns: N/A
        """
        self.feature_column = feature_column
        self.feature_value = feature_value
        self.node_type = node_type
        self.nodes = nodes
        self.children = children
        # End updateTreeValues()

    def isNodeLeaf ( self ):
        """
        Function to determine whether a node is a leaf (i.e., has no children).
            Parameters: 
                self: Indicates class instance
            Returns: N/A
        """
        return self.nodes is None or len ( self.nodes ) == 0
        # End isNodeLeaf
    # End TreeSplits class

# Building Decision Tree
# Base Tree

'''''
Class: BaseTree
    - This class contains all the base functions for a decision tree

Functions:
    -__init__: Initializes the BaseTree class
    - get_valid_midpoints:Function to get the midpoints between values of target array to score in determining 
    best split.
    - get_split_goodness_fit_continuous: Function to evaluate the goodness of the continuous split value.
    - get_min_across_splits_continuous: Function to get the best split (i.e., minimum number of splits)
    across many proposed splits.
    - get_optimal_continuous_feature_split: Function to get the best continuous split for a column.
    - get_discrete_split_value: Function to get the value of making a discrete split.
    - get_optimal_discrete_feature_split: Function to get the best split value for a discrete columns.
    - get_terminal_node: Function to create a terminal node.
    - get_continuous_node: Function to create a continuous node split.
    - get_discrete_node: Function to create a discrete node split.
    - get_next_split: Function get get the next split in the decision tree.
    - fit:Function to fit the decision tree.
    - collect_children: Function to get all the target values of leaves for a subtree. Used for post-pruning.
    - predict_from_all_children: Gets the prediction by treating a subtree as a leaf.
    - predict_node: Function which makes predictions based on a subtree.
    - predict: Function to make predictions over an X matrix.

'''''
class BaseTree:
    """
    Base class with decision tree functionality.
    """

    def __init__ (
        # Initialize parameters
        self,
        map_column_node_type: Dict, # Maps column index to node type
        evaluate_function: Callable, # Takes set of target values & returns score
        agg_function: Callable, # Prediction based on values in a leaf node
        early_stopping_value: float = None,
        early_stopping_comparison: Callable = operator.le,
    ):
        """
        Parameters:
            map_column_node_type: Indicates the mapping from the column index to the node type 
                ("discrete" or "continuous"). This dictates the splitting technique used (Type: dict).
            evaluate_function: Takes a set of target values and returns a score. Seeks to maximize the 
                value. (Type: Callable)
            agg_function: Function used for prediction based on the values in a leaf node. (Type: Callable)
            early_stopping_value: The necessary change in the loss function for the algorithm to continue 
                splitting. Defaults to zero if none is passed. (Type: float)
            early_stopping_comparison: This function determines whether value must be less then or greater 
                than value.Changes based on regression or classification. Defaults to less than or equal to. 
                (Type: Callable)
        """
        self.agg_function = agg_function # For predictions based on values in a leaf
        self.map_column_node_type = map_column_node_type # Map column index to node type
        self.evaluate_function = evaluate_function # Return maximized score
        # Value used to continue splitting
        self.early_stopping_value = (
            0 if not early_stopping_value else early_stopping_value
        ) # Stops at zero if none is passed
        self.n_nodes = 1 # Number of nodes
        self.early_stopping_comparison = early_stopping_comparison
        # End __initi__()

    @staticmethod
    def get_valid_midpoints ( feature_array: np.ndarray, target_array: np.ndarray ):
        """
        Function to get the midpoints between values of the feature array to score in determining best split.
            Parameters:
                feature_array: The feature array ("X") used for splits (Type: Array; np.ndarray)
                target_array: The target array ("Y") (Type: Array; np.ndarray)
            Returns: valid_midpoints: Indicates the points where differences are greater than 0 (uniqueness)
                and y values are not the same
        """
        # Get sorted indices
        indices = np.argsort ( feature_array )

        # Get sorted feature array
        sorted_feature_array = feature_array [ indices ]

        # Get the differences between adjacent values
        feature_array_diffs = np.diff ( sorted_feature_array )

        # Get the midpoints
        midpoints = sorted_feature_array [ 1 : ] - feature_array_diffs / 2

        # Get points where differences are greater than 0 (uniqueness) and y values are not the same
        # See report for details on the latter part.
        valid_midpoints = midpoints [ np.bitwise_and ( feature_array_diffs > 0, np.diff (target_array [ indices ] ) > 0 ) ]

        return valid_midpoints
        # End get_valid_midpoints()

    @staticmethod
    def get_split_goodness_fit_continuous (
        feature_array: np.ndarray, target_array: np.ndarray, split: float, evaluate_function: Callable
    ):
        """
        Function to evaluate the goodness of the continuous split value.
            Parameters:
                feature_array: The feature array ("X") used for splits (Type: Array; np.ndarray)
                target_array: The target array ("Y") (Type: Array; np.ndarray)
                split: The value to split on (Type: Float)
                evaluate_function: Takes a set of target values and returns a score. Seeks to maximize the 
                value. (Type: Callable)
            Returns: Weighted sum of evaluate_function across splits & the gain ratio denominator
        """
        # Get above and below the split value
        above = feature_array >= split
        below = feature_array < split

        # Get weighted average evaluate_function on the splits
        n_above = np.sum ( above )
        above_eval = (
            evaluate_function ( target_array [ above ] ) * n_above / len ( target_array )
        )  # Weight = frac points in above
        below_eval = (
            evaluate_function ( target_array [ below ] ) * ( len ( target_array ) - n_above ) / len ( target_array )
        )  # Weight = frac points not in above

        # returns weighted sum of evaluate_function across splits & the gain ratio denominator
        return (
            above_eval + below_eval,
            -1
            * sum (
                map (
                    lambda x: x * np.log ( x ),
                    [ n_above / len ( target_array ), ( len ( target_array ) - n_above ) / len ( target_array ) ],
                )
            ),
        ) # End get_split_goodness_fit_continuous

    @staticmethod
    def get_min_across_splits_continuous (
        feature_array: np.ndarray, target_array: np.ndarray, splits: np.ndarray, evaluate_function: Callable
    ):
        """
        Function to get the best split (i.e., minimum number) across many proposed splits.
            Parameters:
                feature_array: The feature array ("X") used for splits (Type: Array; np.ndarray)
                target_array: The target array ("Y") (Type: Array; np.ndarray)
                split: The value to split on (Type: Float)
                evaluate_function: Takes a set of target values and returns a score. Seeks to maximize the 
                    value. (Type: Callable)
            Returns: The best splits and the split scores
        """
        n = len ( splits )
        if n > 500:
            # If many split points, use some threading
            with multiprocessing.Pool ( processes = 8 ) as p:
                # Get evaluation scores across all the splits
                post_split_evals = dict (
                    zip (
                        range ( len ( splits ) ),
                        p.starmap (
                            BaseTree.get_split_goodness_fit_continuous,
                            zip ( [ feature_array] * n, [ target_array ] * n, splits, [ evaluate_function ] * n ),
                        ),
                    )
                )
                p.close()
        else:
            # If not too many split points, get scores across all splits
            post_split_evals = dict (
                zip (
                    range ( len ( splits ) ),
                    map (
                        lambda x: BaseTree.get_split_goodness_fit_continuous ( * x ),
                        zip ( [ feature_array ] * n, [ target_array ] * n, splits, [ evaluate_function ] * n ),
                    ),
                )
            )
        # Get the minimum split based on gain ratio
        min_eval = min (
            post_split_evals,
            key = lambda x: pipe (
                post_split_evals.get ( x ),
                lambda results: results [ 0 ] / results [ 1 ],  # entropy / intrinsic value
            ),
        )

        # Return the best split and the splits scores
        return ( splits [ min_eval ], * post_split_evals.get ( min_eval ) )
        # End get_min_across_splits_continuous()

    def get_optimal_continuous_feature_split (
        self, feature_matrix: np.ndarray, target_array: np.ndarray, feature_column: int
    ):
        """
        Function to get the best continuous split for a column
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                feature_column: Indicates the proposed feature column index (Type: Integer)
            Returns: Optimal feature split for continuous data
        """
        midpoints = BaseTree.get_valid_midpoints ( feature_array = feature_matrix [ :, feature_column ], target_array = target_array )
        # If midpoints, get the best one
        if len ( midpoints ) > 0:
            return BaseTree.get_min_across_splits_continuous (
                feature_array = feature_matrix [ :, feature_column ], target_array = target_array, splits = midpoints, evaluate_function = self.evaluate_function
            )

        # If no split points, return inf (can't split here)
        return ( 0, np.inf, 1 )
        # End get_optimal_continuous_feature_split()

    @staticmethod
    def get_discrete_split_value ( feature_array: np.ndarray, target_array: np.ndarray, evaluate_function: Callable ):
        """
        Function to get the value of making a discrete split.
            Parameter:
                feature_array: The feature array used for splits (Type: Array; np.ndarray)
                target_array: The target array ("Y") (Type: Array; np.ndarray)
                evaluate_function: Takes a set of target values and returns a score. Seeks to maximize the 
                    value. (Type: Callable)
            Returns: the weighted average evaluate_function of the split & the intrinsic value to penalize many splits
        """

        # First element is the weighted average evaluate_function of the split
        # Second term is the intrinsic value to penalize many splits.
        return (
            sum (
                [
                    evaluate_function ( target_array [ feature_array == value ] ) * np.sum ( feature_array == value ) / len ( target_array )
                    for value in set ( feature_array )
                ]
            ),
            -1
            * sum (
                [
                    pipe (
                        np.sum ( feature_array == value ) / len ( target_array ),
                        lambda ratio: ratio * np.log ( ratio ),
                    )
                    for value in set ( feature_array )
                ]
            ),
        ) # End get_discrete_split_value()

    def get_optimal_discrete_feature_split (
        self, feature_matrix: np.ndarray, target_array: np.ndarray, feature_column: int
    ):
        """
        Function to get the best split value for a discrete columns
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                feature_column: Indicates the proposed feature column index (Type: Integer)
            Returns: The best split value for discrete data
        """
        return BaseTree.get_discrete_split_value (
            feature_matrix [ :, feature_column ], target_array, evaluate_function = self.evaluate_function
        ) # End get_optimal_discrete_feature_split

    def get_terminal_node (
        self,
        feature_column: int,
        node: TreeSplits,
        feature_value: float,
        feature_matrix: np.ndarray,
        target_array: np.ndarray,
    ):
        """
        Function to create a terminal node.
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                node: The node in the tree to create (Type: TreeSplit object)
                feature_value: The value to split on. None if discrete. (Type: Float)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                feature_column: Indicates the proposed feature column index (Type: Integer)
            Returns: Terminal node
        """
        # Get the node type
        node_type = self.map_column_node_type [ feature_column ]

        if node_type == "continuous":
            # If no feature value is passed, this node is the leaf
            if feature_value is None:
                node.children = target_array
                self.n_nodes += 1
            # If a feature value is passed, create leaves as children
            else:
                # Get the above node
                above = feature_matrix [ :, feature_column ] > feature_value

                # Add two children
                node.updateTreeValues (
                    feature_column = feature_column,
                    feature_value = feature_value,
                    node_type = node_type,
                    nodes = {
                        # Children are above points
                        "above": TreeSplits (
                            children = target_array [ above ]
                        ),  
                        # Children are below points
                        "below": TreeSplits (
                            children = target_array [ np.bitwise_not ( above ) ]
                        ),  
                    },
                )
                # Add two nodes to count
                self.n_nodes += 2
        else:
            # Get the valid values of the discrete column
            unique_x_vals = self.discrete_value_maps [ feature_column ]
            # Create the node
            node.updateTreeValues (
                feature_column = feature_column,
                feature_value = None,
                nodes = {
                    # Add in the matching rows
                    xval: TreeSplits (
                        children = target_array [ feature_matrix [ :, feature_column ] == xval ]
                    )  
                    # If discrete values match
                    if np.any ( feature_matrix [ :, feature_column ] == xval ) 
                    # Add in all the rows if there are no matching values
                    else TreeSplits (
                        children = target_array
                    )  
                    for xval in unique_x_vals
                },
                node_type = "discrete" ,
            )
            # Increment node counter
            self.n_nodes += len ( unique_x_vals )
            # End get_terminal_node()

    def get_continuous_node (
        self,
        feature_column: int,
        feature_value: float,
        feature_matrix: np.ndarray,
        target_array: np.ndarray,
        node: TreeSplits,
    ):
        """
        Function to create a continuous node split.
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                node: The node in the tree to create (Type: TreeSplit object)
                feature_value: The value to split on. None if discrete. (Type: Float)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                feature_column: Indicates the proposed feature column index (Type: Integer)
            Returns: Continuous node
        """
        node.updateTreeValues (
            feature_column = feature_column,
            feature_value = feature_value,
            nodes = { "below": TreeSplits(), "above": TreeSplits() },
            node_type = "continuous",
        )
        # Get the above
        above = feature_matrix [ :, feature_column ] >= feature_value
        # Get the next split for the above node
        self.get_next_split ( feature_matrix = feature_matrix [ above ], target_array = target_array [ above ], tree_split = node.nodes [ "above" ] )
        # Add one node to counter
        self.n_nodes += 1
        # Get the next split for the below node
        self.get_next_split (
            feature_matrix = feature_matrix [ np.bitwise_not ( above ) ],
            target_array = target_array [ np.bitwise_not ( above ) ],
            tree_split = node.nodes [ "below" ],
        )
        # Add one node to counter
        self.n_nodes += 1

        return node # Continuous node
        # End get_continuous_node

    def get_discrete_node ( self, feature_matrix, target_array, feature_column, feature_value, node ):
        """
        Function to create a discrete node split.
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                node: The node in the tree to create (Type: TreeSplit object)
                feature_value: The value to split on. None if discrete. (Type: Float)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                feature_column: Indicates the proposed feature column index (Type: Integer)
            Returns: discrete node
        """
        # Get the unique values for the X poitns
        unique_x_vals = self.discrete_value_maps [ feature_column ]

        # Create the node with an empty child for each x value
        node.updateTreeValues (
            feature_column = feature_column,
            feature_value = feature_value,
            nodes = { xval: TreeSplits() for xval in unique_x_vals },
            node_type = "discrete",
        )

        # For each unique value in the feature column...
        for x_col_value in unique_x_vals:
            # Get the matching rows
            matches = feature_matrix [ :, feature_column ] == x_col_value

            # If no matches, put all points in a leaf node
            if np.sum ( matches ) == 0:
                node.nodes [ x_col_value ] = TreeSplits (
                    node_type = "discrete",
                    feature_column = feature_column,
                    feature_value = x_col_value,
                    children = target_array,
                )
            else:
                # If there are matches, get the next split
                self.get_next_split (
                    feature_matrix = feature_matrix [ matches ], target_array = target_array [ matches ], tree_split = node.nodes [ x_col_value ],
                )
                # Increment by one.
                self.n_nodes += 1

        return node # Discrete nodes
        # End get_discrete_node()

    def get_next_split ( self, feature_matrix: np.ndarray, target_array: np.ndarray, tree_split: TreeSplits):
        """
        Function to get the next split in the decision tree
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
                tree_split: The vertex node to use. This allows the tree to track where to put children
                    (Type: TreeSplit object)
            Returns: The next split in the decision tree

        """
        # If only 1 y value, make a leaf node
        if len ( set ( target_array ) ) == 1:
            tree_split.updateTreeValues (
                feature_column = None,
                feature_value = None,
                node_type = None,
                nodes = {},
                children = target_array,
            )
            return tree_split

        # Get the presplit entropy
        presplit_entropy = self.evaluate_function ( target_array )

        column_values = {}
        for k, v in self.map_column_node_type.items():
            # If there's only one value in feature matrix "X", set the split value to infinity
            if len ( set ( feature_matrix [ :, k ] ) ) == 1:
                value = np.inf
                split = None
                class_ratios = 1
            elif v == "continuous":
                # Get the best possible continuous split for the column
                split, value, class_ratios = self.get_optimal_continuous_feature_split (
                    feature_matrix = feature_matrix, target_array = target_array, feature_column = k
                )
            else:
                # Get the split value for the discrete column
                value, class_ratios = self.get_optimal_discrete_feature_split (
                    feature_matrix = feature_matrix, target_array = target_array, feature_column = k
                )
                split = None

            column_values [ k ] = ( split, value, class_ratios )

        # Get the column with the largest gain ratio
        col_idx_with_min_value = max (
            column_values,
            key = lambda x: ( presplit_entropy - column_values.get ( x ) [ 1 ] )
            / column_values.get ( x ) [ 2 ],
        )

        # If stopping criteria are met or all splits are infinite, terminate the process
        if (
            self.early_stopping_comparison (
                column_values.get ( col_idx_with_min_value ) [ 1 ], self.early_stopping_value
            )
        ) or not np.isfinite ( column_values.get ( col_idx_with_min_value ) [ 1 ] ):
            self.get_terminal_node (
                feature_column = col_idx_with_min_value,
                feature_value = column_values [ col_idx_with_min_value ] [ 0 ],
                node = tree_split,
                feature_matrix = feature_matrix ,
                target_array = target_array,
            )
            return tree_split

        # If the best split is continuous, add a continuous node
        if self.map_column_node_type.get ( col_idx_with_min_value ) == "continuous":
            return self.get_continuous_node (
                feature_column = col_idx_with_min_value,
                feature_value = column_values [col_idx_with_min_value ] [ 0 ],
                feature_matrix = feature_matrix,
                target_array = target_array,
                node = tree_split,
            )

        # Otherwise, add a discrete node.
        else:
            return self.get_discrete_node (
                feature_matrix = feature_matrix,
                target_array = target_array,
                feature_value = column_values [ col_idx_with_min_value ] [ 0 ],
                feature_column = col_idx_with_min_value,
                node = tree_split,
            )
        # End get_next_split

    def fit ( self, feature_matrix: np.ndarray, target_array: np.ndarray):
        """
        Function to fit the decision tree
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
            Returns: N/A
        """
        # Create the root node
        self.root = TreeSplits()

        # Get all possible values for discrete valued columns
        # Necessary so each split can handle unique X values that
        # were not in the training set.
        self.discrete_value_maps = {
            col_idx: np.unique ( feature_matrix [ :, col_idx ] )
            for col_idx, col_type in self.map_column_node_type.items()
            if col_type == "discrete"
        }

        # Start splitting on the root node.
        self.get_next_split ( feature_matrix = feature_matrix, target_array = target_array, tree_split = self.root )
        # End fit()

    @staticmethod
    def collect_children ( node: TreeSplits ):
        """
        Function to get all the target values of leaves for a subtree. Used for post-pruning
            Parameters:
                node: Indicates the root note of the subtree to collect. (Type: TreeSplits object).
            Returns: children nodes
        """
        if node.nodes is None or len ( node.nodes ) == 0:
            return node.children

        # Recursively get all the children and concatenate them
        return np.concatenate (
            [
                BaseTree.collect_children ( child_node )
                for _, child_node in node.nodes.items()
            ]
        ).reshape ( -1 )
        # End collect_children

    def predict_from_all_children ( self, node: TreeSplits ):
        """
        Function gets the prediction by treating a subtree as a leaf.
            Parameter:
                node: Indicates the root node of the subtree. (Type: TreeSplits object)
            Returns: Aggregation of predicted leaf values
        """
        # Collect the children
        children_values = BaseTree.collect_children ( node )
        # Aggregate the leaf values
        return self.agg_function ( children_values )
        # End predict_from_all_children

    def predict_node ( self, feature_matrix: np.ndarray, node: TreeSplits ):
        """
        Function which makes predictions based on a subtree
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                node: Indicates the root node of the subtree. (Type: TreeSplits object)
            Returns: predicted node
        """
        # If leaf, return children target values
        if node.children is not None and len ( node.children ):
            return node.children

        # If continuous, split appropriately, and make recursive call
        if node.node_type == "continuous":
            if feature_matrix [ node.feature_column ] > node.feature_value:
                return self.predict_node ( feature_matrix = feature_matrix, node = node.nodes [ "above" ] )
            else:
                return self.predict_node ( feature_matrix = feature_matrix, node = node.nodes [ "below" ] )

        # If discrete, make recusrive call on node.
        return self.predict_node ( feature_matrix = feature_matrix, node = node.nodes [ feature_matrix [ node.feature_column ] ] )
        # End predict_node()

    def predict ( self, feature_matrix: np.ndarray, node: TreeSplits = None ):
        """
        Function to make predictions over an X matrix
            Parameters:
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                node: This is the node from which to start predictions. If not passed, defaults to the root 
                    node (Type: TreeSplits object)
            Returns: prediction matrix
        """
        node = self.root if not node else node
        if feature_matrix.ndim == 1:
            # If just one row, predict
            return self.agg_function ( self.predict_node ( feature_matrix = feature_matrix, node = node ) )

        # If many rows, map prediction over rows.
        return [ self.agg_function ( self.predict_node ( feature_matrix = row, node = node ) ) for row in feature_matrix ]
        # End predict()
    # End BaseTree class

# Classification Decision Tree (ID3 Algorithm)

'''''
Class: DecisionTreeClassifier
    - This class contains functions to make classification predictions. 
    This is an implementation of the ID3 Algorithm.

Functions:
    -__init__: Initializes the DecisionTreeClassifier class

'''''

class DecisionTreeClassifier ( BaseTree ):
    """
    Class using  mode of leaf nodes for predictions. Works for classification problems
        Parameter:
            BaseTree: Base class with decision tree functionality
    """

    def __init__ ( self, map_column_node_type, evaluate_function, early_stopping_value = None ):
        super().__init__(
            map_column_node_type = map_column_node_type,
            evaluate_function = evaluate_function,
            early_stopping_value = early_stopping_value,
            agg_function = lambda target_array: Counter ( target_array ).most_common ( 1 ) [ 0 ] [ 0 ],  # Get mode
            early_stopping_comparison = operator.le,
        ) # End __init__
    # End DecisionTreeClassifier class

# Regression Decision Tree (CART Algorithm)
'''''
Class: DecisionTreeRegressor
    - This class contains functions to make regression predictions. 
    This is an implementation of the CART algorithm

Functions:
    -__init__: Initializes the DecisionTreeRegressor class

'''''
class DecisionTreeRegressor ( BaseTree ):
    def __init__ ( self, map_column_node_type, evaluate_function, early_stopping_value = None ):
        super().__init__(
            map_column_node_type = map_column_node_type,
            evaluate_function = evaluate_function,
            early_stopping_value = early_stopping_value,
            agg_function = lambda target_array: np.mean ( target_array ),
            early_stopping_comparison = operator.ge,
        ) # End __init__()
    # End DecisionTreeRegressor class

# Pruning Decision Trees
'''''
Class: PostPruner
    - This class runs post-prining with a validation set on a BaseTree

Functions:
    -__init__: Initializes the PostPruner class.
    - tag_node_from_pruning: Function to test a subtree for pruning.
    - prune_node: Prune a given node.
    - prune_tree: Function to prune a tree.
'''''
class PostPruner:

    def __init__(
        # Initialize parameters
        self,
        decision_tree: BaseTree, # Tree to prune
        X_validation: np.ndarray, # The feature matrix of the validation set
        y_validation: np.ndarray, # The target vector of the validation set
        evaluate_function: Callable, # The function to evaluate a split
    ):
        self.evaluate_function = evaluate_function
        self.tree = decision_tree
        self.X_validation = X_validation
        self.y_validation = y_validation
        # End __init__()

    def tag_node_from_pruning ( self, tree, node, feature_matrix, target_array ):
        """
        Function to test a subtree for pruning.
            Parameters:
                tree: Indicates the whole tree to use for predictions (Type: BaseTree object)
                node: This is the node from which to start predictions. If not passed, defaults to the root 
                    node (Type: TreeSplits object)
                feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
                target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
            Returns: True if node is changed; False otherwise
        """
        # If is a leaf, return False
        if node.nodes is None or len ( node.nodes ) == 0:
            return False

        # Score predictions from whole tree
        predictions = tree.predict ( feature_matrix )
        whole_tree_score = self.evaluate_function ( target_array, predictions )

        # Get the children from the node
        children = BaseTree.collect_children ( node )
        # Save original nodes
        original_nodes = node.nodes
        # Update node to be a leaf
        node.updateTreeValues (
            nodes = {},
            children = children,
            feature_column = node.feature_column,
            feature_value = node.feature_value,
            node_type = node.node_type,
        )

        # Score predictions from leaf
        predictions = tree.predict ( feature_matrix )
        pruned_tree_score = self.evaluate_function ( target_array, predictions )

        # If leaf is better, don't swap it back and return True for change
        if whole_tree_score < pruned_tree_score:
            return True

        # Otherwise, change the node back to the original node.
        node.updateTreeValues (
            children = [],
            nodes = original_nodes,
            feature_column = node.feature_column,
            feature_value = node.feature_value,
            node_type = node.node_type,
        )
        # Return False (for no change)
        return False
        #End tag_node_from_pruning()

    def prune_node (self, tree: BaseTree, node: TreeSplits):
        """
        Function to prune a given node
            Parameters:
                tree: Indicates the tree to split over (Type: BaseTree object)
                node: Indicates the node to tag for pruning (Type: TreeSplits object)
            Returns: change_made
        """
        # Prune node, get if change
        change_made = self.tag_node_from_pruning (
            tree = tree, node = node, feature_matrix = self.X_validation, target_array = self.y_validation
        )

        # If change not made and it's not a leaf
        if not change_made and not node.isNodeLeaf():
            # Prune children nodes
            for node_idx, node in node.nodes.items():
                change_made_iter = self.prune_node ( tree = tree, node = node )
                change_made = change_made or change_made_iter  # Track changes
            return change_made

        return change_made
        # End prune_node()

    def prune_tree ( self ):
        """
        Function to prune a tree.
            Parameters:
                self
            Returns: tree
        """
        tree = copy.deepcopy ( self.tree )
        change_made = True
        # As long as changes are made, recursively prune from the root node.
        while change_made:
            change_made = self.prune_node ( tree, tree.root )
        return tree
        # End prune_tree()
    # End PostPruner class

# Functions for data set experiments
# Classification experiment, ID3 Algorithm using Gain-Ratio As Splitting Criterion

def run_classification_experiment ( feature_matrix, target_array, colmap ):
    """
    Function to run classification experiment
        Parameters:
            feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
            target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
            colmap: Indicates the mapping from the column index to the feature type, either 
                "discrete" or "continuous". (Type: Dict)
        Returns: experiment_results
    """
    np.random.seed ( 7062020 ) # Due date

    # Split  off validation set and cross-validation set
    X_validation = feature_matrix [ : feature_matrix.shape [ 0 ] // 10 ]
    X_cross_validation = feature_matrix [ feature_matrix.shape [ 0 ] // 10 : ]
    y_validation = target_array [ : feature_matrix.shape [ 0 ] // 10 ]
    y_cross_validation = target_array [ feature_matrix.shape [ 0 ] // 10 : ]

    experiment_results = {}
    experiment_num = 1

    # Use 5-Fold stratified CV
    kfold_strat = KFoldStratifiedCV ( number_of_folds = 5, shuffle = True )

    for train, test in kfold_strat.split ( feature_matrix = X_cross_validation, target_array = y_cross_validation ):
        logger.info ( f"Experiment Number: { experiment_num }" )

        # Get training set
        X_train = X_cross_validation [ train, : ]
        y_train = y_cross_validation [ train ]

        # Fit the tree
        d_tree = DecisionTreeClassifier ( evaluate_function = entropy, map_column_node_type = colmap )
        d_tree.fit ( X_train, y_train )

        # Prune the tree
        pruned_tree = PostPruner (
            d_tree,
            X_validation = X_validation,
            y_validation = y_validation,
            evaluate_function = accuracy,
        ).prune_tree()

        # Get post-pruned predictions
        pruned_preds = pruned_tree.predict ( X_cross_validation [ test, : ] )

        # Save the results
        experiment_results [ experiment_num ] = {
            "actuals": y_cross_validation [ test ],
            "preds": pruned_preds,
            "model": pruned_tree,
        }
        experiment_num += 1

    return experiment_results
    # End run_classification_experiment

# Functions for data set experiments
# Regression experiment, CART Algorithm using mse as splitting criterion

def run_regression_experiment ( feature_matrix, target_array, early_stopping_values ):
    """
    Function to run regression experiment
        Parameters:
            feature_matrix: Indicates the feature matrix, "X" (Type: Array; np.ndarray)
            target_array: Indicates the target vector, "y" (Type: Array; np.ndarray)
            early_stopping_values: Indicates the iterable set of early stopping values for the experiment. (Type: Float)
        Returns: experiment_results
    """
    np.random.seed ( 7062020 ) # Due date
    X_validation = feature_matrix [ : feature_matrix.shape [ 0 ] // 10 ]
    X_cross_validation = feature_matrix [ feature_matrix.shape [ 0 ] // 10 : ]
    y_validation = target_array [ : feature_matrix.shape [ 0 ] // 10 ]
    y_cross_validation = target_array [ feature_matrix.shape [ 0 ] // 10 : ]

    # Only binary splits in a CART tree.
    colmap = { i: "continuous" for i in range ( X_validation.shape [ 1 ] ) }

    experiment_results = {}
    experiment_num = 1

    kfold = KFoldCV ( number_of_folds = 5, shuffle = True )

    for train, test in kfold.split ( feature_matrix = X_cross_validation, target_array = y_cross_validation ):
        model_callable = partial (
            DecisionTreeRegressor, evaluate_function = mse, map_column_node_type = colmap
        )

        # Get the optimal value of the early stopping parameter
        if experiment_num == 1:
            grid_search_tuner = GridSearchCV (
                param_grid = { "early_stopping_value": early_stopping_values },
                model_callable = model_callable,
                scoring_func = mean_squared_error,
                X_validation = X_validation,
                y_validation = y_validation,
            )

            # Get the lowest MSE across the attempts
            scores = list (
                grid_search_tuner.get_cv_scores (
                    X_cross_validation [ train, : ], y_cross_validation [ train ]
                )
            )
            early_stopping_threshold = sorted ( list ( scores ), key = lambda x: x [ 1 ] ) [ 0 ] [ 0 ] [
                "early_stopping_value"
            ]
            logger.info ( f"Early stopping threshold: { early_stopping_threshold } " )

        logger.info ( f"Experiment Number: { experiment_num }" )

        # Get the training split
        X_train = X_cross_validation [ train, : ]
        y_train = y_cross_validation [ train ]

        d_tree = DecisionTreeRegressor (
            evaluate_function = mse,
            map_column_node_type = colmap,
            early_stopping_value = early_stopping_threshold,
        )

        # Fit the tree and get predictions
        d_tree.fit ( X_train, y_train )
        predictions = d_tree.predict ( X_cross_validation [ test, : ] )

        # Store results
        experiment_results [ experiment_num ] = {
            "actuals": y_cross_validation [ test ],
            "preds": predictions,
            "model": d_tree,
        }
        experiment_num += 1

    return experiment_results
    # End run_regression_experiment    


# # Model Evaluation
# 
# Loss functions are used by algorithms to learn the classification models from the data.
# 
# Classification metrics, however, evaluate the classification models themselves. 
# 
# For a binary classification task, where "1" is taken to mean "positive" or "in the class" and "0" is taken to be "negative" or "not in the class", the cases are:
# 
# 1. The true class can be "1" and the model can predict "1". This is a *true positive* or TP.
# 2. The true class can be "1" and the model can predict "0". This is a *false negative* or FN.
# 3. The true class can be "0" and the model can predict "1". This is a *false positive* or FP.
# 4. The true class can be "0" and the model can predict "0". This is a *true negative* or TN.
# 

# ## Training Learners with Cross-Validation
# 
# Fundamental assumption of machine learning:The data that you train your model on must come from the same distribution as the data you hope to apply the model to.
# 
# Cross validation is the process of training learners using one set of data and testing it using a different set.
# 
# Options:
#     - Divide your data into two sets:
#         1. The training set which you use to build the model
#         2. The test(ing) set which you use to evaluate the model. 
#     - kfolds: Yields multiple estimates of evaluation metric
# 
#     
# ### k-fold Cross-Validation
# 
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
# 
# The procedure has a single parameter called k that refers to the number of groups (or folds) that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=5 becoming 5-fold cross-validation.
# 
# 
# The general procedure is as follows:
# - Shuffle the dataset randomly.
# - Split the dataset into k groups (or folds)
# - Save first fold as the validation set & fit the method on the remaining k-1 folds
# - For each unique group:
#     - Take the group as a hold out or test data set
#     - Take the remaining groups as a training data set
# - Fit a model on the training set and evaluate it on the test set
# - Retain the evaluation score and discard the model
# - Summarize the skill of the model using the sample of model evaluation scores
#     - The average of your k recorded errors is called the cross-validation error and will serve as your performance metric for the model
# 
# Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times.
# 
# Below is the visualization of a k-fold validation when k=10.
# 
# Looks like:
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | **Test** | Train | Train | Train | Train | Train | Train | Train | Train | Train |
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| **Test** | Train | Train | Train | Train | Train | Train | Train | Train |
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| Train | **Test** | Train | Train | Train | Train | Train | Train | Train |
# 
# And finally:
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| Train | Train | Train | Train | Train | Train | Train | Train | **Test** |
# 
# ### Stratified k-fold Cross-Validation
# Stratification is the process of rearranging the data so as to ensure that each fold is a good representative of the whole. For example, in a binary classification problem where each class comprises of 50% of the data, it is best to arrange the data such that in every fold, each class comprises of about half the instances.
# 
# For classification problems, one typically uses stratified k-fold cross-validation, in which the folds are selected so that each fold contains roughly the same proportions of class labels.
# 

# In[8]:


# Model Evaluation

# Teaching Learners with Cross-Validation
# k-Folds

'''''
Class: KFoldCV
    - Class to handle K-fold Cross-Validation

Functions:
    - __init__: Initializes the EditedKNN algorithm 
    - get_indices: Obtains indices of length of rows in feature matrix X
    - get_one_split: Given the split indices, obtains one of the splits
    - get_indices_split: Splits the indices by the number of folds
    - split: Creates a generator of train test splits from the feature matrix X
'''''

class KFoldCV:
    
    """
    Class to handle K-Fold Cross-Validation
        Parameters
            number_of_folds : Indicates the number of folds or splits. Type: Integer
            shuffle : If True, rows will be shuffled before the split. Type: Boolean
    """

    def __init__( self, number_of_folds: int, shuffle: bool = True ):
        # Initialize parameters
        # Class instances
        self.number_of_folds = number_of_folds
        self.shuffle = shuffle

    def get_indices ( self, feature_matrix ):
    
        """
        Function obtains indices of length of rows in feature matrix X
            Parameters
                feature_matrix: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
            Returns: Shuffled K-Fold Indices matrix (arranged by row)
        """
       
        # Shuffle if `self.shuffle` is true.
        nrows = feature_matrix.shape [ 0 ]
        return (
            np.random.permutation (
                np.arange ( nrows )
            )  # Shuffle the rows if `self.shuffle`
            if self.shuffle
            else np.arange ( nrows )
        ) # End get_indices()

    def _get_one_split ( split_indices, number_of_split ):
    
        """
        Given the split indices, function obtains one of the training splits
            Parameters
                number_of_folds: Indicates the number of folds or splits. Type: Integer
                split_indices: Indicates array of indices in the training split. Type: Integer
            Returns: number_of_split. Given the split index, obtains the number of split elememnts
        """
    
        # Given the split indices, get the `number_of_split` element of the indices.
        return ( np.delete ( np.concatenate ( split_indices ), split_indices [ number_of_split ] ),  # Drops the test from the train
            split_indices [ number_of_split ],)  # Gets the train
        # End get_one_split

    def _get_indices_split ( indices, number_of_folds ):
    
        """
        Function splits the indices by the number of folds
            Parameters
                indices: Indicates the index of the training/spilt data Type: Integer
                number_of_folds: Indicates the number of folds or splits. Type: Integer
            Returns: array split by indices
        """
        # Split the indicies by the number of folds
        return np.array_split ( indices, indices_or_sections = number_of_folds )
        # End get_indices_split()

    def split ( self, feature_matrix: np.ndarray, target_array: np.ndarray = None ):
    
        """
        Function creates a generator of train/test splits from feature matrix X
            Parameters
                feature_matrix: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
                target_array: Indicates the target vector, "y". (Type: Array; np.ndarray)
            Returns: All but one split as train data (Type: Array) and one split as test data (Type: Array).
        """
        # Split the indices into `number_of_folds` subarray
        indices = self.get_indices ( feature_matrix )
        split_indices = KFoldCV._get_indices_split ( indices = indices, number_of_folds = self.number_of_folds )
        for number_of_split in range ( self.number_of_folds ):
            # Return all but one split as train, and one split as test
            yield KFoldCV._get_one_split ( split_indices, number_of_split = number_of_split )
        # End split()
    # End class KFoldCV

'''''
Class: KFoldStratifiedCV
    - Class to conduct Stratified K-Fold Cross Validation. Ensures the splitting of data into folds is governed by 
    criteria such as ensuring that each fold has the same proportion of observations with a given categorical 
    value, such as the class outcome value.

Functions:
    - __init__: Initializes the KFoldStratifiedCV algorithm 
    - add_split_col: Adds new column called "split"
    - split: Takes an array of classes, and creates train/test splits with proportional examples for each group.
'''''

class KFoldStratifiedCV:
    
    """
    Class to conduct Stratified K-Fold Cross Validation.
        Parameters
            number_of_folds: Indicates the number of folds or splits. Type: Integer
            
    """

    def __init__ ( self, number_of_folds, shuffle = True ):
        # Initialize parameters
        # Class Instances
        self.number_of_folds = number_of_folds
        self.shuffle = shuffle

    def add_split_col ( self, feature_array ):
    
        """
        Function adds new column called "split"
            Parameters
                feature_array: Indicates the feature array
            Returns: New column in dataframe with index & split
            
        """
        feature_array = feature_array if not self.shuffle else np.random.permutation ( feature_array )
        n = len ( feature_array )
        k = int ( np.ceil ( n / self.number_of_folds ) )
        return pd.DataFrame (
            { "index": feature_array, "split": np.tile ( np.arange ( self.number_of_folds ), k )[ 0:n ] , }
        )

    def split ( self, target_array, feature_matrix = None ):
    
        """
        Function takes an array of classes, and creates train/test splits with proportional examples for each group.
            Parameters
                target_array: Indicates the array of class labels. Type: Array (np.array)
            Returns: Dataframe with index values of not cv split & cv split train and test data
        """
        # Make sure y is an array
        target_array = np.array ( target_array ) if isinstance ( target_array, list ) else target_array

        # Groupby y and add integer indices.
        df_with_split = (
            pd.DataFrame ( { "y": target_array, "index": np.arange ( len ( target_array ) ) } )
            .groupby ( "y" ) [ "index" ]
            .apply ( self.add_split_col )  # Add col for split for instance
        )

        # For each fold, get train and test indices (based on col for split)
        for cv_split in np.arange ( self.number_of_folds - 1, -1, -1 ):
            train_bool = df_with_split [ "split" ] != cv_split
            test_bool = ~ train_bool
            # Yield index values of not cv_split and cv_split for train, test
            yield df_with_split [ "index" ].values [ train_bool.values ], df_with_split [
                "index"
            ].values [ test_bool.values ]
            # End split()
    # End class KFoldStratifiedCV


# ## Parameter Tuning
# 
# Parameter tuning is the process to selecting the values for a model’s parameters that maximize the accuracy of the model.
# 
# 
# A machine learning model has two types of parameters:
# 
#     1. Parameters learned through a machine learning model
#     
#     2. Hyper-parameters passed to the machine learning model
# 
# 
# In KNN algorithm, the hyper-parameter is the specified value of k. 
# 
# Normally we randomly set the value for these hyper parameters and see what parameters result in best performance. However randomly selecting the parameters for the algorithm can be exhaustive.
# 
# 
# ## Grid Search
# 
# Instead of randomly selecting the values of the parameters, GridSearch automatically finds the best parameters for a particular model. Grid Search is one such algorithm.
# 
# Grid Search evaluates all the combinations from a list of desired hyper-parameters and reports which combination has the best accuracy.
# 
# ### Process
# 
# Step 1: Set your hyper-parameters ("param_grid" here).
# 
# Step 2: Fit the model. Use k-fold cross-validation internally on selected hyper-parameters. Store model average & accuracy.
# 
# Step 3: Go back to step 1 changing at least 1 hyper-parameter
# 
# Step 4: Select hyperparameter which gives best performance (highest accuracy)
# 
# Note that the search is not done within each fold. Instead, cross-validation is used to evaluate the performance of the model with the current combination of hyperparameters.

# In[9]:


# Model Evaluation
# Parameter Tuning with Grid Search
            

'''''
Class: GridSearchCV
    - Grid Search evaluates all the combinations from a list of desired hyper-parameters and reports 
    which combination has the best accuracy.

Functions:
    - __init__: Initializes the GridSearchCV algorithm 
    - create_param_grid: Creates a mapping of arguments to values to grid search over.
    - get_single_fitting_iteration: Runs a model fit and validation step.
    - get_cv_scores: Runs the grid search across the parameter grid.
'''''

class GridSearchCV:
    
    """
    Class to assist with grid searching over potential parameter values.
        Parameters:
            model_callable: Function that generates a model object. Should take the keys of param_grid as arguments. Type: Callable
            param_grid: Mapping of arguments to potential values. Type: Dictionary
            scoring_func: Takes in y and yhat and returns a score to be maximized. Type: Callable
            cv_object: A CV object that will be used to make validation splits.
            X_validation: Feature matrix ("X") validation set. If not passed, CV is used. (Type: Array; np.ndarrary)
            y_validation: Target vector (y) validation set. If not passed, CV is used. (Type: Array; np.ndarrary)
    """

    def __init__(
        # Initialize parameters
        self,
        model_callable: Callable, # Generates model object; takes keys of param_grid as arguments
        param_grid: Dict, # Mapped arguments to potential values
        scoring_func: Callable, # Score to be maximized
        cv_object: Union [ KFoldCV, KFoldStratifiedCV ] = None,
        X_validation = None,
        y_validation = None,
    ):
        # Class instances
        self.model_callable = model_callable
        self.param_grid = param_grid
        self.scoring_func = scoring_func
        self.cv_object = cv_object
        self.X_val = X_validation
        self.y_val = y_validation

    @staticmethod
    def create_param_grid ( param_grid: Dict ):
        
        """
        Function creates a mapping of arguments to values to grid search over.
            Parameters:
                param_grid: Dictionary of key:value map (arguments to potential values). Type: Dictionary {kwarg: [values]}
        """
        
        return (
            dict ( zip ( param_grid.keys(), instance ) )
            for instance in product ( * param_grid.values() )
        ) # End create_param_grid

    def get_single_fitting_iteration ( self, feature_matrix: np.ndarray, target_array: np.ndarray, model ):
        
        """
        Function runs a model fit and a validation step.
            Parameters:
                feature_matrix: Indicates the feature matrix for training. Type: Array (np.ndarray)
                target_array: Indicates the arget vector for training. Type: Array (np.ndarray)
                model: Indicates model object with a fit and predict method.
            Returns: mean score
        """
        
        scores = []
        
        if self.cv_object:
            # Create train/test splits
            for train, test in self.cv_object.split ( feature_matrix = feature_matrix, target_array = target_array ):
                # Fit the model
                model.fit ( feature_matrix [ train ], target_array [ train ] )
                # Get the predictions
                yhat = model.predict ( feature_matrix [ test ] )
                # Get the scores
                scores.append ( self.scoring_func ( target_array [ test ], yhat ) )
        else:
            model.fit (feature_matrix, target_array )
            yhat = model.predict ( self.X_val )
            scores.append ( self.scoring_func ( self.y_val, yhat ) )

        # Get the average score.
        return np.mean(scores)
        
        # Create train/test splits
        #for train, test in self.cv_object.split ( feature_matrix = feature_matrix, target_array = target_array ):
            # Fit the model
            #model.fit ( feature_matrix [ train ], target_array [ train ] )
            # Get the predictions
            #yhat = model.predict ( feature_matrix [ test ] )
            # Get the scores
            #scores.append ( self.scoring_func ( target_array [ test ], yhat ) )
        # Get the average score.
        #return np.mean ( scores )
    # End get_single_fitting_iteration()

    def get_cv_scores ( self, feature_matrix: np.ndarray, target_array: np.ndarray ):
        
        """
        Function runs the grid search across the parameter grid.
            Parameters:
                feature_matrix: Indicates the feature matrix. Type: Array (np.ndarray)
                target_array: Indicates the target vector. Type: Array (np.ndarray)
        """
        # Create the parameter grid
        param_grid = list ( GridSearchCV.create_param_grid ( self.param_grid ) )

        # Zip the grid to the results from a single fit
        return zip (
            param_grid,
            [
                self.get_single_fitting_iteration (
                    feature_matrix, target_array, model = self.model_callable ( ** param_set )
                )
                for param_set in param_grid
            ],
        ) # End get_cv_scores
    # End class GridSearchCV

# Other Helpfer Functions
# Evaluation Metrics: Accuracy of Predictions

def accuracy ( actuals, predictions ):
    
    """
    Function to get classifier accuracy
    """
    return np.mean ( actuals == predictions )
    # End accuracy()

# Other Helpfer Functions
# Evaluation Metrics: MSE
# Used for Decision Tree Regression (CART Algorithm)

def mean_squared_error ( actuals, predictions ):
    
    """
    Function to get MSE
    """
    return np.mean ( ( actuals - predictions ) ** 2 )
    # End mean_squared_error()

# Other Helpfer Functions
# Choose best value of k

def choose_k (    
    feature_matrix,
    target_array,
    model_call,
    param_grid,
    scoring_func = accuracy,
    cv = KFoldStratifiedCV ( number_of_folds = 3 ),
):
        
    """
    Function to use cross-validation to choose a value of k
        Parameters:
            feature_matrix: Indicates the feature matrix. Type: Array (np.ndarray)
            target_array: Indicates the target vector. Type: Array (np.ndarray)
            model_call: A function that returns a model object. Its arguments must be the keys in param_grid. Type: Callable
            param_grid: A mapping of arguments to values that we want to try. Type: Dictionary (key:value)
            scoring_func: The function that scores the results of a model. This value is maximized.Type: Callable
            cv: The validation object to use for the cross validation.
        Returns: k (the best value for the number of nearest-neighbors)
    """
    grid_search_cv = GridSearchCV (
        model_callable = model_call,
        param_grid = param_grid,
        scoring_func = scoring_func,
        cv_object = cv,
        )
    
    # Get the last sorted value and take k from that values
    return sorted ( list ( grid_search_cv.get_cv_scores ( feature_matrix, target_array ) ), key = lambda x: x [ 1 ] ) [ -1 ][ 0 ][ "k" ]
    # End choose_k()

#ETL, EDA

# Correlations
def correlations ( data, target_array, xs ):
    rs = [] # pearson's r
    rhos = [] # rho
    for x in xs:
        r = stats.pearsonr ( data [ target_array ], data [ x ] ) [ 0 ]
        rs.append ( r )
        rho = stats.spearmanr ( data [ target_array ], data [ x ] ) [ 0 ]
        rhos.append ( rho )
    return pd.DataFrame ( { "feature": xs, "r": rs, "rho": rhos } )
    # End correlations()

# Pair-wise Comparisons

def describe_by_category ( data, numeric, categorical, transpose = False ):
    grouped = data.groupby ( categorical )
    grouped_y = grouped [ numeric ].describe()
    if transpose:
        print( grouped_y.transpose() )
    else:
        print ( grouped_y )
    # End describe_by_category


# # Abalone Data Set
# ## Extract, Transform, Load: Abalone Data
# 
# 
# ### Description
# 
# A data set to predict the age of abalone from physical measurements
# 
# Data obtained from: https://archive.ics.uci.edu/ml/datasets/Abalone
# 
# ### Attribute Information: 8 Attributes (d)
# 
# 1. Sex: Either male (M), female (F), or infant (I). Variable type: Nominal; Data Type: Character
# 2. Length: Longest shell measurement, listed as mm. Variable type: Continuous
# 3. Diameter: Prependiculaer to length, listed as mm. Variable type: Continuous
# 4. Height: With meat in shell, listed as mm. Variable type: Continuous
# 5. Whole weight: Whole abalone, listed in grams. Variable type: Continuous
# 6. Shucked weight: Weight of meat, listed in grams. Variable type: Continuous
# 7. Viscera weight: Gut weight (after bleeding), listed in grams. Variable type: Continuous
# 8. Shell weight: After being dried, listed in grams. Variable type: Continuous
# 
# ### Prediction Value
# 9. Rings: Used to indicate age. +1.5 gives the age in years. Data Type: Integer

# In[10]:


# Log ETL: Abalone Data
logger.info ( "ETL: Abalone Data Set" )
logger.setLevel ( logging.DEBUG )

# Read Abalone Data
# Create dataframe
abalone_data = pd.read_csv (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        header = None,
        names = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ],
    ).sample ( frac = 1, random_state = 7062020 )


# In[11]:


# Confirm data was properly read by examining data frame
abalone_data.info()


# In[12]:


# Look at first few rows of dataframe
abalone_data.head()


# In[13]:


# Verify whether any values are null
abalone_data.isnull().values.any()


# In[14]:


# Again
abalone_data.isna().any()


# ## (Brief) Exploratory Data Analysis: Abalone Data
# 
# ### Single Variables
# 
# Let's look at the summary statistics & Tukey's 5

# In[15]:


# Log EDA: Abalone Data
logger.info ( "EDA: Abalone Data Set" )


# In[16]:


# Descriptive Statistics
abalone_data.describe()


# **Notes**
# 
# Total number of observations: 4177
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 
# If we wanted, we could use this information for each attribute to calculate the following:
#    - Interquartile Range: Q3-Q1
#    - Whisker: 1.5 * IQR (Outliers lie beyond the whisker)

# ## (Brief) Exploratory Data Analysis: Abalone Data
# 
# ### Pair-Wise: Attribute by Class

# In[17]:


# Frequency of classifications
abalone_data [ 'Rings' ].value_counts() # raw counts


# In[18]:


# Plot diagnosis frequencies
sns.countplot ( abalone_data [ 'Rings' ],label = "Count" ) # boxplot


# In[19]:


def describe_by_category ( data, numeric, categorical, transpose = False ):
    grouped = data.groupby ( categorical )
    grouped_y = grouped [ numeric ].describe()
    if transpose:
        print( grouped_y.transpose() )
    else:
        print ( grouped_y )


# In[20]:


# Descriptive Statistics: Describe each variable by 'Rings' (means only)
abalone_data.groupby ( [ 'Rings' ] )[ "Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight" ].mean()


# In[21]:


# Descriptive Statistics: Describe each variable by Rings
abalone_data.groupby ( [ 'Rings' ] )[ "Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight" ].describe()


# In[22]:


boxplot = abalone_data.boxplot ( column = [ 'Length'], by = [ 'Rings' ] )


# In[23]:


boxplot = abalone_data.boxplot ( column = [ "Diameter" ], by = [ 'Rings' ] ) 


# In[24]:


boxplot = abalone_data.boxplot ( column = [ "Height" ], by = [ 'Rings' ] ) 


# In[25]:


boxplot = abalone_data.boxplot ( column = [ "Whole weight" ], by = [ 'Rings' ] )


# In[26]:


boxplot = abalone_data.boxplot ( column = [ "Shucked weight" ], by = [ 'Rings' ] )


# In[27]:


boxplot = abalone_data.boxplot ( column = [ "Viscera weight" ], by = [ 'Rings' ] ) 


# In[28]:


boxplot = abalone_data.boxplot ( column = [ "Shell weight" ], by = [ 'Rings' ] ) 


# In[29]:


# Descriptive Statistics: Sex by Rings
describe_by_category ( abalone_data, "Sex", "Rings", transpose = True )  


# In[30]:


# Descriptive Statistics: Length by Rings
describe_by_category ( abalone_data, "Length", "Rings", transpose = True )


# In[31]:


# Descriptive Statistics: Diameter by Rings
describe_by_category ( abalone_data, "Diameter", "Rings", transpose = True ) 


# In[32]:


# Descriptive Statistics: Height by Rings
describe_by_category ( abalone_data, "Height", "Rings", transpose = True ) 


# In[33]:


# Descriptive Statistics: Whole Weight by Rings
describe_by_category ( abalone_data, "Whole weight", "Rings", transpose = True ) 


# In[34]:


# Descriptive Statistics: Shucked Weight by Rings
describe_by_category ( abalone_data, "Shucked weight", "Rings", transpose = True )


# In[35]:


# Descriptive Statistics: Viscera weight by Rings
describe_by_category ( abalone_data, "Viscera weight", "Rings", transpose = True )


# In[36]:


# Descriptive Statistics: Shell weight by Rings
describe_by_category ( abalone_data, "Shell weight", "Rings", transpose = True )


# ## Decision Tree: Abalone Data
# 
# ### Assign Feature Matrix & Target Vector

# In[37]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X_abalone = abalone_data.drop ( [ "Rings" ], axis = 1 ).values
y_abalone = abalone_data [ "Rings" ].values


# ### Run Classification Experiment
# ### ID3 Algorithm

# In[38]:


# Decision Tree Classification: Abalone Data
# ID3 Algorithm

if __name__ == "__main__":
    import time
    t = time.time()

# Log Experiment: Decision Tree Classification on Abalone Data Using ID3 Algorithm
logger.info ( "Running Abalone Ring Experiment: Decision Tree Classification using ID3 Algorithm" )

# Run Classification Experiment (ID3 Algorithm)
experiment_results = run_classification_experiment (
    feature_matrix = X_abalone,
    target_array = y_abalone,
    colmap = {
            i: "continuous" if i != 0 else "discrete" for i in range ( X_abalone.shape [ 1 ] )
        },
    ) # End experiment

# Log Experiment Accuracy
logger.info ( { k: accuracy ( v [ "actuals" ], v [ "preds" ] ) for k, v in experiment_results.items() } ) 
# End logging accuracy results


# # Car Evaluation Data Set
# ## Extract, Transform, Load: Car Evaluation Data
# 
# 
# ### Description
# 
# A data set to evaluate cars. Besides the target concept (CAR), the model includes three intermediate concepts: PRICE, TECH, COMFORT. Every concept is in the original model related to its lower level descendants by a set of examples
# 
# Data obtained from: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
# 
# ### Attribute Information: 6 Attributes (d)
# 
# 1. buying: Indicates purchase price category (vhigh, high, med, low). Variable Type: Nominal/Categorical
# 2. maint: Indicates maintenance cost (vhigh, high, med, low). Variable Type: Nominal/Categorical
# 3. doors: Indicates number of doors (2, 3, 4, or 5+). Variable Type: Integer/Categorical
# 4. persons: Indicates maximum passenger capacity (2, 4, more). Variable Type: Integer/Categorical
# 5. lug_boot: Indicates size of the luggage boot (small, med, big). Variable Type: Nominal/Categorical
# 6. safety: Indicates safety category of car (low, med, high). Variable Type: Nominal/Categorical
# 
# ### One Class Label
# 7. acceptable (class attribute): Categorizes car as one of the following categories: unacceptable (unacc), acceptable (acc), good, or very good (vgood). Variable Type: Nominal/Categorical  

# In[39]:


# Log ETL: Car Evaluation Data
logger.info ( "ETL: Car Evaluation Data Set" )

# Read Car Evaluation Data
# Create dataframe
car_data = pipe (
    pd.read_csv (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        header = None,
        names = [
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
            "acceptable",
            ],
        )
    ).sample ( frac = 1 )


# In[40]:


# Confirm data was properly read by examining data frame
car_data.info()


# In[41]:


# Look at first few rows of dataframe
car_data.head()


# In[42]:


# Verify whether any values are null
car_data.isnull().values.any()


# In[43]:


# Again
car_data.isna().any()


# ## (Brief) Exploratory Data Analysis: Car Evaluation Data
# 
# ### Single Variables
# 
# Let's look at the summary statistics & Tukey's 5

# In[44]:


# Log EDA: Car Evaluation Data
logger.info ( "EDA: Car Evaluation Data Set" )

# Descriptive Statistics
car_data.describe()


# In[45]:


# Frequency of classifications
car_data [ 'acceptable' ].value_counts() # raw counts


# In[46]:


# Plot diagnosis frequencies
sns.countplot ( car_data [ 'acceptable' ],label = "Count" ) # boxplot


# ## Decision Tree: Car Evaluation Data
# 
# ### Assign Feature Matrix & Target Vector

# In[47]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X_car_data = car_data.drop ( "acceptable", axis = 1 ).values
y_car_data = car_data [ "acceptable" ].values


# ### Run Classification Experiment
# ### ID3 Algorithm

# In[48]:


# Decision Tree Classification: Car Evaluation Data

# Log Experiment: Decision Tree Classification on Car Evaluation Data using ID3 Algorithm
logger.info ( "Running Car Data Experiment: Decision Tree Classification using ID3 Algorithm" )

# Run Classification Experiment (ID3 Algorithm)
car_experiment_results = run_classification_experiment (
    feature_matrix = X_car_data,
    target_array = y_car_data,
    colmap = { i: "discrete" for i in range ( X_car_data.shape [ 1 ] ) },
) # End experiment

# Log Accuracy
logger.info(
    {
        k: accuracy ( v [ "actuals" ], v [ "preds" ] )
        for k, v in car_experiment_results.items()
    }
) # End logging accuracy


# # Image Segmentation Data Set
# ## Extract, Transform, Load: Image Segmentation Data
# 
# ### Description
# 
# [Classification] The instances were drawn randomly from a database of 7 outdoor images. The images
# were handsegmented to create a classification for every pixel.
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
# 
# ### Attribute Information: 19 Attributes (d)
# 
# 1. region-centroid-col: the column of the center pixel of the region. 
# 2. region-centroid-row: the row of the center pixel of the region. 
# 3. region-pixel-count: the number of pixels in a region = 9. 
# 4. short-line-density-5: the results of a line extractoin algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region. 
# 5. short-line-density-2: same as short-line-density-5 but counts lines of high contrast, greater than 5. 
# 6. vedge-mean: measure the contrast of horizontally adjacent pixels in the region. There are 6, the mean and standard deviation are given. This attribute is used as a vertical edge detector. 
# 7. vegde-sd: (see 6) 
# 8. hedge-mean: measures the contrast of vertically adjacent pixels. Used for horizontal line detection. 
# 9. hedge-sd: (see 8). 
# 10. intensity-mean: the average over the region of (R + G + B)/3 
# 11. rawred-mean: the average over the region of the R value. 
# 12. rawblue-mean: the average over the region of the B value. 
# 13. rawgreen-mean: the average over the region of the G value. 
# 14. exred-mean: measure the excess red: (2R - (G + B)) 
# 15. exblue-mean: measure the excess blue: (2B - (G + R)) 
# 16. exgreen-mean: measure the excess green: (2G - (R + B)) 
# 17. value-mean: 3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics) 
# 18. saturatoin-mean: (see 17) 
# 19. hue-mean: (see 17)
# 
# ### One Class Label
# 20. class (class attribute) 

# In[49]:


# Log ETL: Image Segmentation Data
logger.info ( "ETL: Image Segmentation Data Set" )

# Read Image Segmentation Data
# Create dataframe
image_segmentation_data = pipe (
        r.get (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
        ).text.split ( "\n" ),
        lambda lines: pd.read_csv (
            io.StringIO ( "\n".join ( lines [ 5: ] ) ), header = None, names = lines [ 3 ].split ( "," ) ),
        lambda df: df.assign (
            instance_class = lambda df: df.index.to_series().astype ( "category" ).cat.codes
        ), ).sample ( frac = 1 )


# In[50]:


# Confirm data was properly read by examining data frame
image_segmentation_data.info()


# **Notes**
# 
# As expected, we see 20 columns (19 attributes & one class instance). There are 210 entries (n = 210). We see that the attribute/feature REGION-PIXEL-COUNT is an integer, but all other attributes are float type variables. 

# In[51]:


# Look at first few rows of dataframe
image_segmentation_data.head()


# In[52]:


# Verify whether any values are null
image_segmentation_data.isnull().values.any()


# In[53]:


# Again
image_segmentation_data.isna().any()


# ## (Brief) Exploratory Data Analysis: Image Segmentation Data
# 
# ### Single Variables
# 
# Let's look at the summary statistics & Tukey's 5

# In[54]:


# Log EDA: Image Segmentation Data
logger.info ( "EDA: Image Segmentation Data Set" )

# Descriptive Statistics
image_segmentation_data.describe()


# **Notes** 
# 
# Total number of observations: 210 (i.e., n = 210)
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 
# We'll likely want to discretize these attributes by class

# ## (Brief) Exploratory Data Analysis: Image Segmentation Data
# 
# ### Pair-Wise: Attribute by Class

# In[55]:


# Rename column
image_segmentation_data.rename ( columns = { "instance_class":"class" }, inplace = True )


# In[56]:


# Frequency of glass classifications
image_segmentation_data [ 'class' ].value_counts() # raw counts


# **Notes**
# 
# There are 7 image segmentation classifications (labeled 0, 1, 2, 3, .., 7)
# 
# Each image segmentation classification has 30 observations 
# 

# In[57]:


# Plot diagnosos frequencies
sns.countplot ( image_segmentation_data [ 'class' ],label = "Count" ) # boxplot


# **Notes**
# 
# There are 7 image segmentation classifications (labeled 0, 1, 2, 3, .., 7)
# 
# Each image segmentation classification has 30 observations 

# In[58]:


# Get column names
print ( image_segmentation_data.columns )


# In[59]:


# Descriptive Statistics: Describe each variable by class (means only)
image_segmentation_data.groupby ( [ 'class' ] )[ 'REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT',
       'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN',
       'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN', 'RAWRED-MEAN',
       'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN',
       'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN' ].mean()


# In[60]:


# Descriptive Statistics: Describe each variable by class (all variables)
image_segmentation_data.groupby ( [ 'class' ] ) [ 'REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT',
       'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN',
       'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN', 'RAWRED-MEAN',
       'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN',
       'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN' ].describe()        


# In[61]:


boxplot = image_segmentation_data.boxplot ( column = [ 'REGION-CENTROID-COL', 'REGION-CENTROID-ROW' ], by = [ 'class' ] )  


# In[62]:


boxplot = image_segmentation_data.boxplot ( column = [ 'REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5' ], by = [ 'class' ] )


# In[63]:


boxplot = image_segmentation_data.boxplot ( column = [ 'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN' ], by = [ 'class' ] )


# In[64]:


boxplot = image_segmentation_data.boxplot ( column = [ 'VEDGE-SD', 'HEDGE-MEAN' ], by = [ 'class' ] )


# In[65]:


boxplot = image_segmentation_data.boxplot ( column = [ 'HEDGE-SD', 'INTENSITY-MEAN'], by = [ 'class' ] )


# In[66]:


boxplot = image_segmentation_data.boxplot ( column = [ 'RAWRED-MEAN','RAWBLUE-MEAN' ], by = [ 'class' ] )


# In[67]:


boxplot = image_segmentation_data.boxplot ( column = [ 'RAWGREEN-MEAN' ], by = [ 'class' ] )


# In[68]:


boxplot = image_segmentation_data.boxplot ( column = [ 'EXRED-MEAN', 'EXBLUE-MEAN' ], by = [ 'class' ] )


# In[69]:


boxplot = image_segmentation_data.boxplot ( column = [ 'EXGREEN-MEAN'], by = [ 'class' ] )


# In[70]:


boxplot = image_segmentation_data.boxplot ( column = [ 'VALUE-MEAN', 'SATURATION-MEAN' ], by = [ 'class' ] )


# In[71]:


boxplot = image_segmentation_data.boxplot ( column = [ 'HUE-MEAN' ], by = [ 'class' ] )


# In[72]:


# Descriptive Statistics: Describe each variable by class
# REGION-CENTROID-COL by Class
describe_by_category ( image_segmentation_data, 'REGION-CENTROID-COL', "class", transpose = True )


# In[73]:


# Descriptive Statistics: Describe each variable by class
# REGION-CENTROID-ROW by Class
describe_by_category ( image_segmentation_data, 'REGION-CENTROID-ROW', "class", transpose = True )


# In[74]:


# Descriptive Statistics: Describe each variable by class
# REGION-PIXEL-COUNT by Class
describe_by_category ( image_segmentation_data, 'REGION-PIXEL-COUNT', "class", transpose = True )


# In[75]:


# Descriptive Statistics: Describe each variable by class
# SHORT-LINE-DENSITY-5 by Class
describe_by_category ( image_segmentation_data, 'SHORT-LINE-DENSITY-5', "class", transpose = True )


# In[76]:


# Descriptive Statistics: Describe each variable by class
# SHORT-LINE-DENSITY-2 by Class
describe_by_category ( image_segmentation_data, 'SHORT-LINE-DENSITY-2', "class", transpose = True )


# In[77]:


# Descriptive Statistics: Describe each variable by class
# VEDGE-MEAN by Class
describe_by_category ( image_segmentation_data, 'VEDGE-MEAN', "class", transpose = True )


# In[78]:


# Descriptive Statistics: Describe each variable by class
# VEDGE-SD by Class
describe_by_category ( image_segmentation_data, 'VEDGE-SD', "class", transpose = True )


# In[79]:


# Descriptive Statistics: Describe each variable by class
# 'HEDGE-MEAN' by Class
describe_by_category ( image_segmentation_data, 'HEDGE-MEAN', "class", transpose = True )


# In[80]:


# Descriptive Statistics: Describe each variable by class
# HEDGE-SD by Class
describe_by_category ( image_segmentation_data, 'HEDGE-SD', "class", transpose = True )


# In[81]:


# Descriptive Statistics: Describe each variable by class
# INTENSITY-MEAN by Class
describe_by_category ( image_segmentation_data, 'INTENSITY-MEAN', "class", transpose = True )


# In[82]:


# Descriptive Statistics: Describe each variable by class
# RAWRED-MEAN by Class
describe_by_category ( image_segmentation_data, 'RAWRED-MEAN', "class", transpose = True )


# In[83]:


# Descriptive Statistics: Describe each variable by class
# RAWBLUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'RAWBLUE-MEAN', "class", transpose = True )


# In[84]:


# Descriptive Statistics: Describe each variable by class
# RAWGREEN-MEAN by Class
describe_by_category ( image_segmentation_data, 'RAWGREEN-MEAN', "class", transpose = True )


# In[85]:


# Descriptive Statistics: Describe each variable by class
# EXRED-MEAN by Class
describe_by_category ( image_segmentation_data, 'EXRED-MEAN', "class", transpose = True )


# In[86]:


# Descriptive Statistics: Describe each variable by class
# EXBLUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'EXBLUE-MEAN', "class", transpose = True )


# In[87]:


# Descriptive Statistics: Describe each variable by class
# EXGREEN-MEAN by Class
describe_by_category ( image_segmentation_data, 'EXGREEN-MEAN', "class", transpose = True )


# In[88]:


# Descriptive Statistics: Describe each variable by class
# VALUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'VALUE-MEAN', "class", transpose = True )


# In[89]:


# Descriptive Statistics: Describe each variable by class
# SATURATION-MEAN by Class
describe_by_category ( image_segmentation_data, 'SATURATION-MEAN', "class", transpose = True )


# In[90]:


# Descriptive Statistics: Describe each variable by class
# HUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'HUE-MEAN', "class", transpose = True )


# ## Decision Tree: Image Segmentation Data
# 
# ### Assign Feature Matrix & Target Vector

# In[91]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X_image_seg = image_segmentation_data.drop ( [ "class", "REGION-PIXEL-COUNT"], axis = 1 ).values
y_image_seg = image_segmentation_data [ "class" ].values


# ### Run Classification Experiment
# ### ID3 Algorithm

# In[92]:


# Decision Tree Classification: Image Segmentation Data

# Log Experiment: Decision Tree Classification on Image Segmentation Data Using ID3 Algorithm
logger.info ( "Running Image Segmentation Experiment: Decision Tree Classification using ID3 Algorithm" )

# Run Classification Experiment Using ID3 Algorithm
image_seg_experiment_results = run_classification_experiment (
    feature_matrix = X_image_seg,
    target_array = y_image_seg,
    colmap = { i: "continuous" for i in range ( X_image_seg.shape [ 1 ] ) },
    ) # End experiment

# Log Experiment's Accuracy
logger.info(
    {
        k: accuracy ( v [ "actuals" ], v [ "preds" ] )
        for k, v in image_seg_experiment_results.items()
    }
) # End Logging Accuracy


# # Wine Quality Data Set
# ## Extract, Transform, Load: Wine Quality Data
# 
# ### Description
# 
# The goal is to model wine quality based on physicochemical tests. Can be used for classification or regression.
# The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones)
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Wine+Quality
# 
# ### Attribute Information: 11 Attributes (d) based on physiochemnical tests
# 
# 1. fixed acidity
# 2. volatile acidity 
# 3. citric acid 
# 4. residual sugar
# 5. chlorides
# 6. free sulfur dioxide 
# 7. total sulfur dioxide 
# 8. density
# 9. pH 
# 10. sulphates 
# 11. alcohol 
# Output variable (based on sensory data): 
# 12 - quality (score between 0 and 10)
# 
# ### One Class Label (based on sensory data):
# 12. Quality (score between 0 and 10).

# In[93]:


# Log ETL: Wine Quality Data
logger.info ( "ETL: Wine Quality Data Set" )

# Read Wine Quality Data
# Create dataframe & label columns
white_wine_data = pd.read_csv (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    sep = ";",
)

red_wine_data = pd.read_csv (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    sep = ";",
)

# Concat datasets and create an indicator for the wine type.
wine_data = pd.concat ( [ white_wine_data.assign ( is_white = 1 ), red_wine_data.assign ( is_white = 0 ) ] ).sample ( frac = 1 )


# In[94]:


# Confirm data was properly read by examining data frame
wine_data.info()


# In[95]:


# Verify whether any values are null
wine_data.isnull().values.any()


# In[96]:


# Again
wine_data.isna().any()


# In[97]:


# Look at first few rows of dataframe
wine_data.head()


# In[98]:


# Get column names
print ( wine_data.columns )


# ## (Brief) Exploratory Data Analysis: Wine Quality Data
# 
# ### Single Variable

# In[99]:


# Log EDA: Wine Quality Data
logger.info ( "EDA: Wine Quality Data Set" )

# Descriptive Statistics
wine_data.describe()


# ## (Brief) Exploratory Data Analysis: Wine Quality Data
# 
# ### Pair-Wise: Attribute by Class

# In[100]:


# Frequency of diagnoses classifications
wine_data [ 'quality' ].value_counts() # raw counts


# In[101]:


# Plot diagnosos frequencies
sns.countplot ( wine_data [ 'quality' ],label = "Count" ) # boxplot


# ## Decision Tree: Wine Quality Data
# 
# ### Assign Feature Matrix & Target Vector

# In[102]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X_wine_data = wine_data.drop ( "quality", axis = 1 ).values
y_wine_data = wine_data [ "quality" ].values


# ### Run Regression Experiment
# ### CART Algorithm

# In[103]:


# Decision Tree Regression: Wine Quality Data using CART Algorithm

# Log Experiment: Decision Tree Regression on Wine Quality Data using CART
logger.info ( "Running Wine Quality Experiment: Decision Tree Regression using CART Algorithm" )

# Run Regression Experiment using CART Algorithm
wine_experiment_results = run_regression_experiment (
    feature_matrix = X_wine_data, 
    target_array = y_wine_data, 
    early_stopping_values = np.linspace ( 0.2, 1, 4 )
) # End experiment

# Log Experiment Accuracy
logger.info (
    {
        k: mean_squared_error ( v [ "actuals" ], v [ "preds" ] )
        for k, v in wine_experiment_results.items()
    }
) # End logging accuracy


# # Computer Hardware Data Set
# ## Extract, Transform, Load: Computer Hardware Data
# 
# ### Description
# 
# [Regression] The estimated relative performance values were estimated by the authors using a linear
# regression method. 
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Iris
# 
# ### Attribute Information: 9 Attributes (d)
# 
# 2. Model Name: many unique symbols 
# 3. MYCT: machine cycle time in nanoseconds (integer) 
# 4. MMIN: minimum main memory in kilobytes (integer) 
# 5. MMAX: maximum main memory in kilobytes (integer) 
# 6. CACH: cache memory in kilobytes (integer) 
# 7. CHMIN: minimum channels in units (integer) 
# 8. CHMAX: maximum channels in units (integer) 
# 9. PRP: published relative performance (integer) 
# 10. ERP: estimated relative performance from the original article (integer)
# 
# ### One Class Label
# 1. Vendor Name (class):
#     - adviser
#     - amdah
#     - apollo
#     - basf
#     - bti
#     - burroughs
#     - c.r.d
#     - cambex
#     - cdc
#     - dec 
#     - dg
#     - formation
#     - four-phase
#     - gould
#     - honeywell
#     - hp
#     - ibm
#     - ipl
#     - magnuson
#     - microdata
#     - nas
#     - ncr
#     - nixdorf
#     - perkin-elmer
#     - prime
#     - siemens
#     - sperry
#     - sratus
#     - wang

# In[104]:


# Log ETL: Computer Hardware Data
logger.info ( "ETL: Computer Hardware Data Set" )

# Read Computer Hardware Data
# Create dataframe & label columns
computer_hardware_data = pd.read_csv (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data",
        header = None,
        names = [
            "vendor_name",
            "model_name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP",
            "ERP",
        ],
    )


# In[105]:


# Confirm data was properly read by examining data frame
computer_hardware_data.info()


# **Notes**
# 
# As expected, we see 10 columns (9 attributes and 1 class label). There are 209 entries (n = 209). We see that the instance class (vendor_name) is an object, as is the model_name, but all other attributes are integer type variables.

# In[106]:


# Verify whether any values are null
computer_hardware_data.isnull().values.any()


# **Notes**
# 
# We observe no null instances

# In[107]:


# Again
computer_hardware_data.isna().any()


# **Notes**
# 
# We observe no null instances in any of the attribute columns

# In[108]:


# Look at first few rows of dataframe
computer_hardware_data.head()


# In[109]:


# Classification for Class Label: data frame for this category
computer_hardware_data[ "vendor_name" ].astype ( "category" ).cat.codes


# ## (Brief) Exploratory Data Analysis: Computer Hardware Data
# 
# ### Single Variable

# In[110]:


# Log EDA: Computer Hardware Data
logger.info ( "EDA: Computer Hardware Data Set" )

# Descriptive Statistics
computer_hardware_data.describe()


# **Notes**
# 
# Total number of observations: 209
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 

# ## (Brief) Exploratory Data Analysis: Computer Hardware Data
# 
# ### Pair-Wise: Attribute by Class

# In[111]:


# Frequency of diagnoses classifications
computer_hardware_data [ 'vendor_name' ].value_counts() # raw counts


# In[112]:


# Plot diagnosos frequencies
sns.countplot ( computer_hardware_data [ 'vendor_name' ],label = "Count" ) # boxplot


# In[113]:


# Descriptive Statistics: Describe each variable by class (means only)
computer_hardware_data.groupby ( [ 'vendor_name' ] )[ "MYCT", "MMIN", "MMAX", "CACH","CHMIN", "CHMAX","PRP", "ERP" ].mean()


# In[114]:


# Descriptive Statistics: Describe each variable by class (means only)
computer_hardware_data.groupby ( [ 'vendor_name' ] )[ "MYCT", "MMIN", "MMAX", "CACH","CHMIN", "CHMAX","PRP", "ERP" ].describe()


# In[115]:


boxplot = computer_hardware_data.boxplot ( column = [ "MYCT" ], by = [ 'vendor_name' ] )


# In[116]:


boxplot = computer_hardware_data.boxplot ( column = [ "CACH" ], by = [ 'vendor_name' ] )


# In[117]:


boxplot = computer_hardware_data.boxplot ( column = [ "MMIN" ], by = [ 'vendor_name' ] )


# In[118]:


boxplot = computer_hardware_data.boxplot ( column = [ "MMAX" ], by = [ 'vendor_name' ] )


# In[119]:


boxplot = computer_hardware_data.boxplot ( column = [ "CHMIN" ], by = [ 'vendor_name' ] )


# In[120]:


boxplot = computer_hardware_data.boxplot ( column = [ "CHMAX" ], by = [ 'vendor_name' ] )


# In[121]:


boxplot = computer_hardware_data.boxplot ( column = [ "PRP" ], by = [ 'vendor_name' ] )


# In[122]:


boxplot = computer_hardware_data.boxplot ( column = [ "ERP" ], by = [ 'vendor_name' ] )


# In[123]:


# Descriptive Statistics: Attribute by Class
# MYCT by Class
describe_by_category ( computer_hardware_data, "MYCT", "vendor_name", transpose = True )


# **Notes**
# 

# In[124]:


# Descriptive Statistics: Attribute by Class
# MMIN by Class
describe_by_category ( computer_hardware_data, "MMIN", "vendor_name", transpose = True )


# **Notes**
# 
# 

# In[125]:


# Descriptive Statistics: Attribute by Class
# MMAX by Class
describe_by_category ( computer_hardware_data, "MMAX", "vendor_name", transpose = True )


# **Notes**
# 

# In[126]:


# Descriptive Statistics: Attribute by Class
# CACH by Class
describe_by_category ( computer_hardware_data, "CACH", "vendor_name", transpose = True )


# **Notes**
# 
# 

# In[127]:


# Descriptive Statistics: Attribute by Class
# CHMIN by Class
describe_by_category ( computer_hardware_data, "CHMIN", "vendor_name", transpose = True )


# In[128]:


# Descriptive Statistics: Attribute by Class
# CHMAX by Class
describe_by_category ( computer_hardware_data, "CHMAX", "vendor_name", transpose = True )


# In[129]:


# Descriptive Statistics: Attribute by Class
# PRP by Class
describe_by_category ( computer_hardware_data, "PRP", "vendor_name", transpose = True )


# In[130]:


# Descriptive Statistics: Attribute by Class
# ERP by Class
describe_by_category ( computer_hardware_data, "ERP", "vendor_name", transpose = True )


# ## Decision Tree: Computer Hardware Data
# 
# ### Assign Feature Matrix & Target Vector
# 

# In[131]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X_cpu = computer_hardware_data.drop ( [ "vendor_name", "model_name", "PRP", "ERP", "MYCT" ], axis = 1 ).values
y_cpu = computer_hardware_data [ "PRP" ].values


# ### Run Regression Experiment
# ### CART Algorithm

# In[132]:


# Decision Tree Regression: Computer Hardware Data using CART Algorithm

# Log Experiment: Decision Tree Regression on Computer Hardware Data using CART Algorithm
logger.info ( "Running Experiment on Computer Hardware Data: Decision Tree Regression using CART Algorithm" )

# Run Regression Experiment using CART Algorithm
computer_hardware_experiment_results = run_regression_experiment (
    feature_matrix = X_cpu, 
    target_array = y_cpu, 
    early_stopping_values = np.linspace ( 1500, 30000, 1000 )
) # End experiment

# Log Experiment Accuracy    
logger.info (
    {
        k: mean_squared_error ( v [ "actuals" ], v [ "preds" ] )
        for k, v in computer_hardware_experiment_results.items()
    }
) # End logging accuracy


# # Forest Fires Data Set
# ## Extract, Transform, Load: Forest Fires Data
# 
# ### Description
# 
# [Regression] This is a difficult regression task, where the aim is to predict the burned area of forest
# fires, in the northeast region of Portugal, by using meteorological and other data .
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Forest+Fires
# 
# ### Attribute Information: 13 Attributes (d)
# 
# 1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9 
# 2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9 
# 3. month - month of the year: 'jan' to 'dec' 
# 4. day - day of the week: 'mon' to 'sun' 
# 5. FFMC - FFMC index from the FWI system: 18.7 to 96.20 
# 6. DMC - DMC index from the FWI system: 1.1 to 291.3 
# 7. DC - DC index from the FWI system: 7.9 to 860.6 
# 8. ISI - ISI index from the FWI system: 0.0 to 56.10 
# 9. temp - temperature in Celsius degrees: 2.2 to 33.30 
# 10. RH - relative humidity in %: 15.0 to 100 
# 11. wind - wind speed in km/h: 0.40 to 9.40 
# 12. rain - outside rain in mm/m2 : 0.0 to 6.4 
# 13. area - the burned area of the forest (in ha): 0.00 to 1090.84 
# (this output variable is very skewed towards 0.0, thus it may make 
# sense to model with the logarithm transform).
# 

# In[133]:


# Log ETL: Forest Fire Data
logger.info ( "ETL: Forest Fire Data Set" )

# Read Forest Fire Data
# Create dataframe
forest_fire_data = pd.read_csv (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    )


# In[134]:


# Confirm data was properly read by examining data frame
forest_fire_data.info()


# **Notes**
# 
# As expected, we see 13 columns (# attributes and # class label). There are 517 entries (n = 517). We see that month and day attributes are objects; X, Y, RH are integer type variables; and FFMC, DMC, DC, ISI, temp, wind, rain, area variables are all float type.

# In[135]:


# Verify whether any values are null
forest_fire_data.isnull().values.any()


# **Note**
# 
# We see there are no null instances

# In[136]:


# Again
forest_fire_data.isna().any()


# ## (Brief) Exploratory Data Analysis: Forrest Fire Data
# 
# ### Single Variable
# 
# Let's look at the summary statistics & Tukey's 5
# 

# In[137]:


# Look at first few rows of dataframe
forest_fire_data.head()


# ## (Brief) Exploratory Data Analysis: Forest Fire Data
# 
# ### Single Variable
# 
# Let's look at the summary statistics & Tukey's 5
# 

# In[138]:


# Log EDA: Forest Fire Data
logger.info ( "EDA: Forest Fire Data Set" )

# Descriptive Statistics
forest_fire_data.describe()


# **Notes**
# 
# Total number of observations: 517
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 

# ## (Brief) Exploratory Data Analysis: Forest Fire Data
# 
# ### Pair-Wise: Attribute by Class

# In[139]:


# Frequency of diagnoses classifications
forest_fire_data [ 'area' ].value_counts() # raw counts


# In[140]:


# Get Columns
list ( forest_fire_data.columns )


# In[141]:


# Descriptive Statistics: Describe each variable by class (means only)
forest_fire_data.groupby ( [ 'area' ] )[ 'X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain' ].mean()


# In[142]:


# Descriptive Statistics: Describe each variable by class (means only)
forest_fire_data.groupby ( [ 'area' ] )[ 'X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain' ].describe()


# In[143]:


boxplot = forest_fire_data.boxplot ( column = [ "X", "Y"], by = [ 'area' ] )


# In[144]:


boxplot = forest_fire_data.boxplot ( column = [ 'FFMC','DMC' ], by = [ 'area' ] )


# In[145]:


boxplot = forest_fire_data.boxplot ( column = [ "DC", "ISI" ], by = [ 'area' ] )


# In[146]:


boxplot = forest_fire_data.boxplot ( column = [ "temp", "RH" ], by = [ 'area' ] )


# In[147]:


boxplot = forest_fire_data.boxplot ( column = [ "wind", "rain" ], by = [ 'area' ] )


# ## Decision Tree: Forest Fire Data
# 
# ### Assign Feature Matrix & Target Vector

# In[148]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X_fires = ( forest_fire_data.drop ( "area", axis = 1 ).pipe ( lambda df: pd.get_dummies ( df, columns = [ "month", "day" ], drop_first = True ) ).values )
y_fires = forest_fire_data [ "area" ].values


# ### Run Regression Experiment
# ### CART Algorithm

# In[149]:


# Decision Tree Regression: Forest Fire Data using CART Algorithm

# Log Experiment: Decision Tree Regression on Forest Fire Data using CART Algorithm
logger.info ( "Running Experiment on Forest Fire Data: Decision Tree Regression Using CART Algorithm" )

# Run Regression Experiment using CART Algorithm
fires_experiment = run_regression_experiment (
    feature_matrix = X_fires,
    target_array = y_fires,
    early_stopping_values = np.linspace ( 1875, 5000, 500 )
) # End experiment

# Log Accuracy
logger.info (
    {
        k: mean_squared_error ( v [ "actuals" ], v [ "preds" ] )
        for k, v in fires_experiment.items()
    }
) # End logging accuracy

logger.info ( f"Run time: { time.time() }" )


# In[ ]:




