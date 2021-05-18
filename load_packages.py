#load libraries
import numpy
import sklearn
import math
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
#import pandasql as ps
from pandas import set_option
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
#from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from os import makedirs
from keras.models import load_model
from numpy import dstack
from keras.layers.merge import concatenate
from numpy import argmax
from keras.models import Model
from keras.utils import plot_model