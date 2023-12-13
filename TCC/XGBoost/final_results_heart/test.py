# %% [markdown]
# ## Definindo o Problema

# %%
#!pip install lightgbm;
#!pip install pymoo;
#!pip install -U pyrecorder;

# %%
import numpy as np
def special_floor(x):
    x = int(np.round(x))
    if x == 0:
        x = 1
    return x

# %%
ITERATIONS = 32
POPULATION = 32

# %%
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from yellowbrick.classifier import ConfusionMatrix
import numpy as np
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
#from google.colab import drive
import pickle
from pymoo.algorithms.soo.nonconvex.pso import PSO, PSOAnimation
#from pymoo.factory import Rastrigin
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from pymoo.factory import get_termination
#from pymoo.util.display import Display
from pymoo.core.callback import Callback

# %%
#!pip install XGBoost

# %%
import xgboost as xgb

# %%
#drive.mount('/content/gdrive')

# %%
#with open('/content/gdrive/MyDrive/datasets/credit.pkl', 'rb') as f:
#    x_heart_train, x_heart_test, y_heart_train, y_heart_test = pickle.load(f)
    
#with open('./heart.pkl', 'rb') as f:
#    x_heart, y_heart = pickle.load(f)
    
with open('./heart.pkl', 'rb') as f:
    x_heart, y_heart = pickle.load(f)

# %% [markdown]
# Definindo as restrições 

# %%
DIMENSIONS = 16

# %%

n_estimators_max = 1000
learning_rate_max = 0.6
subsample_max = 1.0
colsample_bytree_max = 1.0
gamma_max = 0.5
max_depth_max = 10
min_child_weight_max = 10
reg_alpha_max = 0.1
reg_lambda_max = 1
scale_pos_weight_max = 10
base_score_max = 1





n_estimators_min = 10
learning_rate_min = 0.0001
subsample_min = 0.6
colsample_bytree_min = 0.6
gamma_min = 0
max_depth_min = 3
min_child_weight_min = 1
reg_alpha_min = 0
reg_lambda_min = 0
scale_pos_weight_min = 1
base_score_min = 0


# %% [markdown]
# Definindo o problema

# %%
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class OptimizeWithAccuracy(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var= 11,
                         n_obj=1,
                         n_constr=11,
                         xl=np.array([
                          n_estimators_min,
                            learning_rate_min,
                            subsample_min,
                            colsample_bytree_min,
                            gamma_min,
                            max_depth_min,
                            min_child_weight_min,
                            reg_alpha_min,
                            reg_lambda_min,
                            scale_pos_weight_min,
                            base_score_min
                          ]),
                         xu=np.array([
                            n_estimators_max,
                            learning_rate_max,
                            subsample_max,
                            colsample_bytree_max,
                            gamma_max,
                            max_depth_max,
                            min_child_weight_max,
                            reg_alpha_max,
                            reg_lambda_max,
                            scale_pos_weight_max,
                            base_score_max,
                            ])
                        )

    def _evaluate(self, x, out, *args, **kwargs):
        #num_leaves, min_child_samples, n_estimators, learning_rate, subsample_for_bin, min_split_gain, min_child_weight, reg_alpha, reg_lambda
       
        model_xgboost = xgb.XGBClassifier(
                          n_estimators = int(np.round(x[0])),
                          learning_rate = x[1],
                          subsample = x[2],
                          colsample_bytree = x[3],
                          gamma = x[4],
                          max_depth = special_floor(x[5]),
                          min_child_weight = int(np.round(x[6])),
                          reg_alpha = x[7],
                          reg_lambda = x[8],
                          scale_pos_weight = int(x[9]),
                          base_score       = x[10],
                          n_jobs = -1
                                       )
        
        kfold = KFold(n_splits = 3, shuffle = True)
        
        scores = cross_val_score(model_xgboost, x_heart, y_heart, cv = kfold, n_jobs = -1)  
        result = scores.mean()
        #if not result:
        #  result = 0
        
        out['F'] = -1 * result

problemAccuracy = OptimizeWithAccuracy()

# %%
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class OptimizeWithF1(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var= 11,
                         n_obj=1,
                         n_constr=11,
                         xl=np.array([
                          n_estimators_min,
                            learning_rate_min,
                            subsample_min,
                            colsample_bytree_min,
                            gamma_min,
                            max_depth_min,
                            min_child_weight_min,
                            reg_alpha_min,
                            reg_lambda_min,
                            scale_pos_weight_min,
                            base_score_min
                          ]),
                         xu=np.array([
                            n_estimators_max,
                            learning_rate_max,
                            subsample_max,
                            colsample_bytree_max,
                            gamma_max,
                            max_depth_max,
                            min_child_weight_max,
                            reg_alpha_max,
                            reg_lambda_max,
                            scale_pos_weight_max,
                            base_score_max,
                            ])
                        )

    def _evaluate(self, x, out, *args, **kwargs):
        #num_leaves, min_child_samples, n_estimators, learning_rate, subsample_for_bin, min_split_gain, min_child_weight, reg_alpha, reg_lambda
       
        model_xgboost = xgb.XGBClassifier(
                          n_estimators = int(np.round(x[0])),
                          learning_rate = x[1],
                          subsample = x[2],
                          colsample_bytree = x[3],
                          gamma = x[4],
                          max_depth = special_floor(x[5]),
                          min_child_weight = int(np.round(x[6])),
                          reg_alpha = x[7],
                          reg_lambda = x[8],
                          scale_pos_weight = int(x[9]),
                          base_score       = x[10],
                          n_jobs = -1
                                       )
        
        kfold = KFold(n_splits = 3, shuffle = True)

        scores = cross_val_score(model_xgboost, x_heart, y_heart, cv = kfold, scoring='f1', n_jobs = -1)  
        result = scores.mean()
        out['F'] = -1 * result

problemF1 = OptimizeWithF1()

# %%
xgb_mean = []
for i in range(30):
    model_xgboost = xgb.XGBClassifier(
                        n_jobs = -1
                                    )
    

    kfold = KFold(n_splits = 3, shuffle = True)

    scores = cross_val_score(model_xgboost, x_heart, y_heart, cv = kfold, scoring='f1', n_jobs = -1)  

    result = scores.mean()
    
    xgb_mean.append(result)
    
xgb_mean# Convert the Python array to a numpy array
numpy_array = np.array(xgb_mean)

# Calculate the average value using numpy.mean()
average_value = np.mean(numpy_array)

print("'", average_value, "'")

# %%
import numpy as np
from pymoo.core.problem import ElementwiseProblem

class OptimizeWithAUC(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var= 11,
                         n_obj=1,
                         n_constr=11,
                         xl=np.array([
                          n_estimators_min,
                            learning_rate_min,
                            subsample_min,
                            colsample_bytree_min,
                            gamma_min,
                            max_depth_min,
                            min_child_weight_min,
                            reg_alpha_min,
                            reg_lambda_min,
                            scale_pos_weight_min,
                            base_score_min
                          ]),
                         xu=np.array([
                            n_estimators_max,
                            learning_rate_max,
                            subsample_max,
                            colsample_bytree_max,
                            gamma_max,
                            max_depth_max,
                            min_child_weight_max,
                            reg_alpha_max,
                            reg_lambda_max,
                            scale_pos_weight_max,
                            base_score_max,
                            ])
                        )

    def _evaluate(self, x, out, *args, **kwargs):
        #num_leaves, min_child_samples, n_estimators, learning_rate, subsample_for_bin, min_split_gain, min_child_weight, reg_alpha, reg_lambda
       
        model_xgboost = xgb.XGBClassifier(
                          n_estimators = int(np.round(x[0])),
                          learning_rate = x[1],
                          subsample = x[2],
                          colsample_bytree = x[3],
                          gamma = x[4],
                          max_depth = special_floor(x[5]),
                          min_child_weight = int(np.round(x[6])),
                          reg_alpha = x[7],
                          reg_lambda = x[8],
                          scale_pos_weight = int(x[9]),
                          base_score       = x[10],
                          n_jobs = -1
                                       )

        
        kfold = KFold(n_splits = 3, shuffle = True)
  
        scores = cross_val_score(model_xgboost, x_heart, y_heart, cv = kfold, scoring='roc_auc', n_jobs = -1)  
        
        result = scores.mean()

        out['F'] = -1 * result

problemAUC = OptimizeWithAUC()

# %%
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

# %%
class MyOutput(Output):

    def __init__(self):
        super().__init__()
        global pbar 
        pbar = tqdm(total=ITERATIONS)
        self.score = Column("score", width=13)
        self.Parameters = Column("Parameters", width=35)
        self.columns += [self.score, self.Parameters]

    def update(self, algorithm):
        super().update(algorithm)
        self.score.set(-np.min(algorithm.pop.get("F")))
        #self.Parameters.set(algorithm.pop.get("X")[0])
        pbar.update(1)
        if pbar.n == ITERATIONS: pbar.close()

# %% [markdown]
# ## Particle Swarm Optimization (PSO)

# %% [markdown]
# ### Acurácia

# %%
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np

# %%
xl=np.array([n_estimators_min,
             learning_rate_min,
             subsample_min,
             colsample_bytree_min,
             gamma_min,
             max_depth_min,
             min_child_weight_min,
             reg_alpha_min,
             reg_lambda_min,
             scale_pos_weight_min,
             base_score_min])
xu=np.array([n_estimators_max,
             learning_rate_max,
             subsample_max,
             colsample_bytree_max,
             gamma_max,
             max_depth_max,
             min_child_weight_max,
             reg_alpha_max,
             reg_lambda_max,
             scale_pos_weight_max,
             base_score_max])

# %%
def PSO_Optimize_Accuracy(values):
    x = values[0] 
    model_xgboost = xgb.XGBClassifier(
                          n_estimators = int(np.round(x[0])),
                          learning_rate = x[1],
                          subsample = x[2],
                          colsample_bytree = x[3],
                          gamma = x[4],
                          max_depth = special_floor(x[5]),
                          min_child_weight = int(np.round(x[6])),
                          reg_alpha = x[7],
                          reg_lambda = x[8],
                          scale_pos_weight = int(x[9]),
                          base_score       = x[10],
                          n_jobs = -1
                                    )
    
    kfold = KFold(n_splits = 10, shuffle = True)
    
    scores = cross_val_score(model_xgboost, x_heart, y_heart, cv = kfold, n_jobs = -1)  
    result = scores.mean()     

    return -result

# %%
def PSO_Optimize_F1(values):
    x = values[0] 
    model_xgboost = xgb.XGBClassifier(
                          n_estimators = int(np.round(x[0])),
                          learning_rate = x[1],
                          subsample = x[2],
                          colsample_bytree = x[3],
                          gamma = x[4],
                          max_depth = special_floor(x[5]),
                          min_child_weight = int(np.round(x[6])),
                          reg_alpha = x[7],
                          reg_lambda = x[8],
                          scale_pos_weight = int(x[9]),
                          base_score       = x[10],
                          n_jobs = -1
                                    )
    
    kfold = KFold(n_splits = 10, shuffle = True)
    
    scores = cross_val_score(model_xgboost,  x_heart, y_heart, cv = kfold, n_jobs = -1, scoring='f1')  
    result = scores.mean()     

    return -result

# %%
def PSO_Optimize_AUC(values):
    x = values[0] 
    model_xgboost = xgb.XGBClassifier(
                          n_estimators = int(np.round(x[0])),
                          learning_rate = x[1],
                          subsample = x[2],
                          colsample_bytree = x[3],
                          gamma = x[4],
                          max_depth = special_floor(x[5]),
                          min_child_weight = int(np.round(x[6])),
                          reg_alpha = x[7],
                          reg_lambda = x[8],
                          scale_pos_weight = int(x[9]),
                          base_score       = x[10],
                          n_jobs = -1
                                    )
    
    kfold = KFold(n_splits = 10, shuffle = True)
    
    scores = cross_val_score(model_xgboost, x_heart, y_heart, cv = kfold, n_jobs = -1, scoring='roc_auc')  
    result = scores.mean()     

    return -result

# %%
def run_accuracy_pso():
    # Call an instance of PSO
    swarm_size = 32
    iters = 32
    dim = 11
    options = {'c1': 1.5, 'c2':1.5, 'w':0.5}
    constraints = (xl,xu)

    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                        dimensions=dim,
                                        options=options,
                                        bounds=constraints)
    cost, joint_vars = optimizer.optimize(objective_func = PSO_Optimize_Accuracy, iters=iters)
    return -cost

# %%
def run_auc_pso():
    # Call an instance of PSO
    swarm_size = 32
    iters = 32
    dim = 11
    options = {'c1': 1.5, 'c2':1.5, 'w':0.5}
    constraints = (xl,xu)

    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                        dimensions=dim,
                                        options=options,
                                        bounds=constraints)
    cost, joint_vars = optimizer.optimize(objective_func = PSO_Optimize_AUC, iters=iters)
    return -cost

# %%
def run_f1_pso():
    # Call an instance of PSO
    swarm_size = 32
    iters = 32
    dim = 11
    options = {'c1': 1.5, 'c2':1.5, 'w':0.5}
    constraints = (xl,xu)

    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                        dimensions=dim,
                                        options=options,
                                        bounds=constraints)
    cost, joint_vars = optimizer.optimize(objective_func = PSO_Optimize_F1, iters=iters)
    return -cost

# %% [markdown]
# ## Algoritmo Genético (GA)

# %% [markdown]
# ### Acurácia

# %%
ITERATIONS = 32
POPULATION = 32

# %%
from pymoo.algorithms.soo.nonconvex.ga import GA

# %%
def run_accuracy_ga(ITERATIONS = 32, POPULATION = 32):

    algorithm = GA(pop_size=POPULATION)

    term = get_termination("n_gen", ITERATIONS)

    res = minimize(problemAccuracy,
                algorithm,
                save_history=True,
                verbose=True,
                output=MyOutput(),
                termination = term)


    index_best_individual = np.where(res.pop.get('F') == np.min(res.pop.get('F')))[0][0]
    score_best_individual = res.pop.get('F')[index_best_individual]
    parameters_best_individual = res.pop.get('X')[index_best_individual]

    #print(f'Best Accuracy Score {-score_best_individual}')
    #print(f'Model parameters: \n {parameters_best_individual}')
    
    return score_best_individual, parameters_best_individual, res

# %%
#tracking = [-np.min(individual.pop.get('F')) for individual in res.history ]
#tracking_GA_Accuracy = tracking
#plt.plot(tracking)

# %%
#trlist = np.array([])
#for i in range(len(res.history)):
#  trlist = np.append(trlist, -res.history[i].pop.get('F').reshape(-1)) 

# %%
#scorelist = []
#
#for i in range(len(trlist)):
#  if i == 0:
#    scorelist.append(trlist[i])
#  elif trlist[i] > scorelist[i-1]:
#    scorelist.append(trlist[i])
#  else:
#    scorelist.append(scorelist[i-1])

# %%
#Accuracy_GA = scorelist

# %%
#plt.plot(scorelist);

# %% [markdown]
# ### F1 Score 

# %%
from pymoo.algorithms.soo.nonconvex.ga import GA
def run_f1_ga(ITERATIONS = 32, POPULATION = 32):
    algorithm = GA(pop_size=POPULATION)

    term = get_termination("n_gen", ITERATIONS)

    res = minimize(problemF1,
                algorithm,
                save_history=True,
                verbose=False,
                output=MyOutput(),
                termination = term)


    index_best_individual = np.where(res.pop.get('F') == np.min(res.pop.get('F')))[0][0]
    score_best_individual = res.pop.get('F')[index_best_individual]
    parameters_best_individual = res.pop.get('X')[index_best_individual]

    #print(f'Best F1 Score {-score_best_individual}')
    #print(f'Model parameters: \n {parameters_best_individual}')
    
    return score_best_individual, parameters_best_individual, res

# %%
#trlist = np.array([])
#for i in range(len(res.history)):
#  trlist = np.append(trlist, -res.history[i].pop.get('F').reshape(-1)) 
#  
#scorelist = []
#
#for i in range(len(trlist)):
##for i in range(1):
#  if i == 0:
#    scorelist.append(trlist[i])
#  elif trlist[i] > scorelist[i-1]:
#    scorelist.append(trlist[i])
#  else:
#    scorelist.append(scorelist[i-1])
#    
#F1_GA = scorelist
#
#plt.plot(scorelist);

# %%
#tracking = [-np.min(individual.pop.get('F')) for individual in res.history ]
#tracking_GA_F1 = tracking
#plt.plot(tracking)

# %% [markdown]
# ### AUC

# %%
from pymoo.algorithms.soo.nonconvex.ga import GA
def run_auc_ga(ITERATIONS = 32, POPULATION = 32):
    algorithm = GA(pop_size=POPULATION)

    term = get_termination("n_gen", ITERATIONS)

    res = minimize(problemAUC,
                algorithm,
                save_history=False,
                verbose=False,
                output=MyOutput(),
                termination = term)

    
    index_best_individual = np.where(res.pop.get('F') == np.min(res.pop.get('F')))[0][0]
    score_best_individual = res.pop.get('F')[index_best_individual]
    parameters_best_individual = res.pop.get('X')[index_best_individual]

    #print(f'Best AUC Score {-score_best_individual}')
    #print(f'Model parameters: \n {parameters_best_individual}')
    
    return score_best_individual, parameters_best_individual, res

# %%
#trlist = np.array([])
#for i in range(len(res.history)):
#  trlist = np.append(trlist, -res.history[i].pop.get('F').reshape(-1)) 
#  
#scorelist = []
#
#for i in range(len(trlist)):
##for i in range(1):
#  if i == 0:
#    scorelist.append(trlist[i])
#  elif trlist[i] > scorelist[i-1]:
#    scorelist.append(trlist[i])
#  else:
#    scorelist.append(scorelist[i-1])
#    
#AUC_GA = scorelist
#
#plt.plot(scorelist);

# %%
#tracking = [-np.min(individual.pop.get('F')) for individual in res.history ]
#tracking_GA_AUC = tracking
#plt.plot(tracking)

# %% [markdown]
# ## Grid Search

# %% [markdown]
# ### Acurácia

# %%
from sklearn.model_selection import GridSearchCV

# %%
#n_possibilities = 2
#
##num_leaves_grid = [i for i in range(num_leaves_min,num_leaves_max, int((num_leaves_max)/13))]
#num_leaves_grid = [i for i in map(lambda x: int(x), np.linspace(num_leaves_min, num_leaves_max, n_possibilities))]
#num_leaves_grid = num_leaves_grid + [100, 50, 75, 125, 11,150]
##print('Num_Leaves_Grid: ')
##print(num_leaves_grid, len(num_leaves_grid))
##print('\n')
#
#
#min_child_samples_grid = [i for i in map(lambda x: int(x), np.linspace(min_child_samples_min, min_child_samples_max, n_possibilities))]
##print('min_child_samples_grid:')
##print(min_child_samples_grid, len(min_child_samples_grid))
##print('\n')
#
#n_estimators_grid = [i for i in map(lambda x: int(x), np.linspace(n_estimators_min, n_estimators_max, n_possibilities))]
##print('n_estimators_grid:')
##print(n_estimators_grid, len(n_estimators_grid))
##print('\n')
#
#learning_rate_grid = np.linspace(learning_rate_min, learning_rate_max, n_possibilities)
##print('learning_rate_grid:')
##print(learning_rate_grid, len(learning_rate_grid))
##print('\n')
#
#subsample_for_bin_grid = [i for i in map(lambda x: int(x), np.linspace(subsample_for_bin_min, subsample_for_bin_max, n_possibilities))]
##print('subsample_for_bin_grid:')
##print(subsample_for_bin_grid, len(subsample_for_bin_grid))
##print('\n')
#
#min_split_gain_grid = np.linspace(min_split_gain_min, min_split_gain_max, n_possibilities)
##print('min_split_gain_grid:')
##print(min_split_gain_grid, len(min_split_gain_grid))
##print('\n')
#
#min_child_weight_grid = np.linspace(min_child_weight_min, min_child_weight_max, n_possibilities)
##print('min_child_weight_grid:')
##print(min_child_weight_grid, len(min_child_weight_grid))
##print('\n')
#
#reg_alpha_grid = np.linspace(reg_alpha_min, reg_alpha_max, n_possibilities)
##print('reg_alpha_grid:')
##print(reg_alpha_grid, len(reg_alpha_grid))
##print('\n')


# %%
#parametros = {'num_leaves': num_leaves_grid, #int
#              'min_child_samples': min_child_samples_grid,#int
#              'n_estimators': n_estimators_grid, #int
#              'learning_rate': learning_rate_grid,
#              'subsample_for_bin': subsample_for_bin_grid, # int
#              'min_split_gain': min_split_gain_grid,
#              'min_child_weight': min_child_weight_grid,
#              'reg_alpha': reg_alpha_grid,
#              'max_depth': [-1],
#              'n_jobs': [-1]}

# %%
#kfold = KFold(n_splits = 3, shuffle = True)
#grid_search = GridSearchCV(estimator = lgb.LGBMClassifier(), param_grid = parametros, cv = kfold, n_jobs= -1, verbose = 3)
#grid_search.fit(x_heart, y_heart)
#melhores_parametros = grid_search.best_params_
#melhor_resultado = grid_search.best_score_
##print(melhores_parametros)
##print(melhor_resultado)

# %%
#grid_search.cv_results_.keys()

# %%
#trlist = grid_search.cv_results_['mean_test_score']

# %%
#scorelist = []
#
#for i in range(len(trlist)):
##for i in range(1):
#  if i == 0:
#    scorelist.append(trlist[i])
#  elif trlist[i] > scorelist[i-1]:
#    scorelist.append(trlist[i])
#  else:
#    scorelist.append(scorelist[i-1])

# %%
#tracking_GS_Accuracy = scorelist
#plt.plot(scorelist); 

# %% [markdown]
# ### F1 Score

# %%
#kfold = KFold(n_splits = 3, shuffle = True)
#grid_search = GridSearchCV(estimator = lgb.LGBMClassifier(), param_grid = parametros, cv = kfold, n_jobs= -1, scoring='f1', verbose = 3)
#grid_search.fit(x_heart, y_heart)
#melhores_parametros = grid_search.best_params_
#melhor_resultado = grid_search.best_score_
##print(melhores_parametros)
##print(melhor_resultado)
#
#trlist = grid_search.cv_results_['mean_test_score']
#
#
#scorelist = []
#
#for i in range(len(trlist)):
##for i in range(1):
#  if i == 0:
#    scorelist.append(trlist[i])
#  elif trlist[i] > scorelist[i-1]:
#    scorelist.append(trlist[i])
#  else:
#    scorelist.append(scorelist[i-1])
#    
#tracking_GS_F1 = scorelist
#plt.plot(scorelist); 

# %% [markdown]
# ### AUC

# %%
#kfold = KFold(n_splits = 3, shuffle = True)
#grid_search = GridSearchCV(estimator = lgb.LGBMClassifier(), param_grid = parametros, cv = kfold, n_jobs= -1, scoring='roc_auc', verbose = 3)
#grid_search.fit(x_heart, y_heart)
#melhores_parametros = grid_search.best_params_
#melhor_resultado = grid_search.best_score_
##print(melhores_parametros)
##print(melhor_resultado)
#
#trlist = grid_search.cv_results_['mean_test_score']
#
#
#scorelist = []
#
#for i in range(len(trlist)):
##for i in range(1):
#  if i == 0:
#    scorelist.append(trlist[i])
#  elif trlist[i] > scorelist[i-1]:
#    scorelist.append(trlist[i])
#  else:
#    scorelist.append(scorelist[i-1])
#    
#tracking_GS_AUC = scorelist
#plt.plot(scorelist); 

# %% [markdown]
# ## Optuna

# %% [markdown]
# ### Acurácia

# %%
# %%
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import sklearn
from sklearn import datasets
def objective_accuracy(trial):
      
      n_estimators = trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max)
      learning_rate = trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max)
      subsample = trial.suggest_float('subsample', subsample_min, subsample_max)
      colsample_bytree = trial.suggest_float('colsample_bytree', colsample_bytree_min, colsample_bytree_max)
      gamma = trial.suggest_float('gamma', gamma_min, gamma_max)
      max_depth = trial.suggest_int('max_depth', max_depth_min, max_depth_max)
      min_child_weight = trial.suggest_int('min_child_weight', min_child_weight_min, min_child_weight_max)
      reg_alpha = trial.suggest_float('reg_alpha', reg_alpha_min, reg_alpha_max)
      reg_lambda = trial.suggest_float('reg_lambda', reg_lambda_min, reg_lambda_max)
      scale_pos_weight = trial.suggest_int('scale_pos_weight', scale_pos_weight_min, scale_pos_weight_max)
      base_score = trial.suggest_float('base_score', base_score_min, base_score_max)
      
      model_xgboost = xgb.XGBClassifier(
                        n_estimators = n_estimators,
                        learning_rate = learning_rate,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree,
                        gamma = gamma,
                        max_depth = max_depth,
                        min_child_weight = min_child_weight,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        scale_pos_weight = scale_pos_weight,
                        base_score = base_score,
                        n_jobs = -1
                                    )

      
      kfold = KFold(n_splits = 3, shuffle = True)
      
      return sklearn.model_selection.cross_val_score(model_xgboost, x_heart, y_heart, n_jobs = -1, cv=kfold).mean()

# %%
def run_optuna_accuracy(n_trials=1024):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_accuracy, n_trials=n_trials, n_jobs = -1)
    trial = study.best_trial
    #print('Accuracy: {}'.format(trial.value))
    #print("Best hyperparameters: {}".format(trial.params))
    return trial.value, study


# %%
trial, study = run_optuna_accuracy(1)

# %%
trial

# %%
#tracking = []
#for i in range(len(study.trials)):
#  if i == 0: 
#    tracking.append(study.trials[i].value)
#  elif tracking[i-1] > study.trials[i].value:
#    tracking.append(tracking[i-1])
#  else:
#    tracking.append(study.trials[i].value)
##tracking_sorted = sorted(tracking)

# %%
#tracking_Optuna_Accuracy = tracking

# %%
#plt.plot(tracking);
#plt.plot(tracking_sorted) 

# %%
#optuna.visualization.plot_optimization_history(study)

# %%
#optuna.visualization.plot_slice(study)

# %% [markdown]
# ### F1 Score

# %%
def objective_f1(trial):
  
      n_estimators = trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max)
      learning_rate = trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max)
      subsample = trial.suggest_float('subsample', subsample_min, subsample_max)
      colsample_bytree = trial.suggest_float('colsample_bytree', colsample_bytree_min, colsample_bytree_max)
      gamma = trial.suggest_float('gamma', gamma_min, gamma_max)
      max_depth = trial.suggest_int('max_depth', max_depth_min, max_depth_max)
      min_child_weight = trial.suggest_int('min_child_weight', min_child_weight_min, min_child_weight_max)
      reg_alpha = trial.suggest_float('reg_alpha', reg_alpha_min, reg_alpha_max)
      reg_lambda = trial.suggest_float('reg_lambda', reg_lambda_min, reg_lambda_max)
      scale_pos_weight = trial.suggest_int('scale_pos_weight', scale_pos_weight_min, scale_pos_weight_max)
      base_score = trial.suggest_float('base_score', base_score_min, base_score_max)
      
      model_xgboost = xgb.XGBClassifier(
                        n_estimators = n_estimators,
                        learning_rate = learning_rate,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree,
                        gamma = gamma,
                        max_depth = max_depth,
                        min_child_weight = min_child_weight,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        scale_pos_weight = scale_pos_weight,
                        base_score = base_score,
                        n_jobs = -1
                                    )

      
      kfold = KFold(n_splits = 3, shuffle = True)

      return sklearn.model_selection.cross_val_score(model_xgboost, x_heart, y_heart, n_jobs = -1, scoring='f1', cv=kfold).mean()

# %%
def run_optuna_f1(n_trials=1024):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_f1, n_trials=n_trials, n_jobs = -1)
    trial = study.best_trial
    #print('F1: {}'.format(trial.value))
    #print("Best hyperparameters: {}".format(trial.params))
    return trial.value, study


# %%
#tracking = []
#for i in range(len(study.trials)):
#  if i == 0: 
#    tracking.append(study.trials[i].value)
#  elif tracking[i-1] > study.trials[i].value:
#    tracking.append(tracking[i-1])
#  else:
#    tracking.append(study.trials[i].value)
##tracking_sorted = sorted(tracking)
#
#tracking_Optuna_F1 = tracking
#
#plt.plot(tracking);
##plt.plot(tracking_sorted) 

# %% [markdown]
# ### AUC

# %%
def objective_auc(trial):
  
      n_estimators = trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max)
      learning_rate = trial.suggest_float('learning_rate', learning_rate_min, learning_rate_max)
      subsample = trial.suggest_float('subsample', subsample_min, subsample_max)
      colsample_bytree = trial.suggest_float('colsample_bytree', colsample_bytree_min, colsample_bytree_max)
      gamma = trial.suggest_float('gamma', gamma_min, gamma_max)
      max_depth = trial.suggest_int('max_depth', max_depth_min, max_depth_max)
      min_child_weight = trial.suggest_int('min_child_weight', min_child_weight_min, min_child_weight_max)
      reg_alpha = trial.suggest_float('reg_alpha', reg_alpha_min, reg_alpha_max)
      reg_lambda = trial.suggest_float('reg_lambda', reg_lambda_min, reg_lambda_max)
      scale_pos_weight = trial.suggest_int('scale_pos_weight', scale_pos_weight_min, scale_pos_weight_max)
      base_score = trial.suggest_float('base_score', base_score_min, base_score_max)
      
      model_xgboost = xgb.XGBClassifier(
                        n_estimators = n_estimators,
                        learning_rate = learning_rate,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree,
                        gamma = gamma,
                        max_depth = max_depth,
                        min_child_weight = min_child_weight,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        scale_pos_weight = scale_pos_weight,
                        base_score = base_score,
                        n_jobs = -1
                                    )

      kfold = KFold(n_splits = 3, shuffle = True)


      return sklearn.model_selection.cross_val_score(model_xgboost, x_heart, y_heart, n_jobs = -1, scoring='roc_auc', cv=kfold).mean()

# %%
def run_optuna_auc(n_trials=1024):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_auc, n_trials=n_trials, n_jobs = -1)
    trial = study.best_trial
    #print('AUC: {}'.format(trial.value))
    #print("Best hyperparameters: {}".format(trial.params))
    return trial.value, study


# %%
#tracking = []
#for i in range(len(study.trials)):
#  if i == 0: 
#    tracking.append(study.trials[i].value)
#  elif tracking[i-1] > study.trials[i].value:
#    tracking.append(tracking[i-1])
#  else:
#    tracking.append(study.trials[i].value)
##tracking_sorted = sorted(tracking)
#
#tracking_Optuna_AUC = tracking
#
#plt.plot(tracking);
##plt.plot(tracking_sorted) 

# %% [markdown]
# # Análise Comparativa

# %%
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
#
#ax1.set_title('PSO X Gerações')
#ax2.set_title('PSO X Avaliações')
#
#ax1.plot(tracking_PSO_Accuracy)
#ax2.plot(Accuracy_PSO)
#plt.show()

# %%
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
#
#ax1.set_title('GA X Gerações')
#ax2.set_title('GA X Avaliações')
#
#ax1.plot(tracking_GA_Accuracy)
#ax2.plot(Accuracy_GA)
#plt.show()

# %% [markdown]
# Gráficos Comparativos Acurácia

# %%
#from IPython.core.pylabtools import figsize
#from matplotlib.pyplot import figure
#import matplotlib.patches as mpatches
#
#plt.figure(figsize=(12, 6))
#red_patch = mpatches.Patch(color='red', label='PSO')
#blue_patch = mpatches.Patch(color='blue', label='GA')
#green_patch = mpatches.Patch(color='green', label='Optuna')
##black_patch = mpatches.Patch(color='black', label='GridSearch')
#
#plt.legend(handles=[red_patch, blue_patch, green_patch])
#plt.plot(Accuracy_GA, color = 'blue', linewidth=1, linestyle='-', )
#plt.plot(Accuracy_PSO, color = 'red', linewidth=1, linestyle='-')
#plt.plot(tracking_Optuna_Accuracy, color = 'green', linewidth=1, linestyle='-')
##plt.plot(tracking_GS, color = 'black', linewidth=3, linestyle='-')
#plt.show()

# %%
#fig, ax = plt.subplots(2, 2, figsize=(25,15))
#
#ax[0, 0].set_title(f'Genetic Algorithm (GA) - Accuracy: {max(tracking_GA_Accuracy)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#ax[0, 1].set_title(f'Particle Swarm Optimization (PSO) - Accuracy: {max(tracking_PSO_Accuracy)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#
#ax[1, 0].set_title(f'Grid Search - Accuracy: {max(tracking_GS_Accuracy)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#ax[1, 1].set_title(f'Optuna - Accuracy: {max(tracking_Optuna_Accuracy)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#
#ax[0,0].plot(tracking_GA_Accuracy, color = 'blue', linewidth=3, linestyle='-', )
#ax[0,1].plot(tracking_PSO_Accuracy, color = 'red', linewidth=3, linestyle='-')
#
#ax[1,0].plot(tracking_GS_Accuracy, color = 'black', linewidth=3, linestyle='-')
#ax[1,1].plot(tracking_Optuna_Accuracy, color = 'green', linewidth=3, linestyle='-')


# %% [markdown]
# F1 score

# %%
#plt.figure(figsize=(12, 6))
#red_patch = mpatches.Patch(color='red', label='PSO')
#blue_patch = mpatches.Patch(color='blue', label='GA')
#green_patch = mpatches.Patch(color='green', label='Optuna')
##black_patch = mpatches.Patch(color='black', label='GridSearch')
#
#plt.legend(handles=[red_patch, blue_patch, green_patch])
#plt.plot(F1_GA, color = 'blue', linewidth=1, linestyle='-', )
#plt.plot(F1_PSO, color = 'red', linewidth=1, linestyle='-')
#plt.plot(tracking_Optuna_F1, color = 'green', linewidth=1, linestyle='-')
##plt.plot(tracking_GS, color = 'black', linewidth=3, linestyle='-')
#plt.show()

# %%
#fig, ax = plt.subplots(2, 2, figsize=(25,15))
#
#ax[0, 0].set_title(f'Genetic Algorithm (GA) - Accuracy: {max(tracking_GA_F1)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#ax[0, 1].set_title(f'Particle Swarm Optimization (PSO) - Accuracy: {max(tracking_PSO_F1)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#
#ax[1, 0].set_title(f'Grid Search - Accuracy: {max(tracking_GS_F1)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#ax[1, 1].set_title(f'Optuna - Accuracy: {max(tracking_Optuna_F1)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#
#ax[0,0].plot(tracking_GA_F1, color = 'blue', linewidth=3, linestyle='-', )
#ax[0,1].plot(tracking_PSO_F1, color = 'red', linewidth=3, linestyle='-')
#
#ax[1,0].plot(tracking_GS_F1, color = 'black', linewidth=3, linestyle='-')
#ax[1,1].plot(tracking_Optuna_F1, color = 'green', linewidth=3, linestyle='-')


# %% [markdown]
# ROC AUC

# %%
#plt.figure(figsize=(12, 6))
#red_patch = mpatches.Patch(color='red', label='PSO')
#blue_patch = mpatches.Patch(color='blue', label='GA')
#green_patch = mpatches.Patch(color='green', label='Optuna')
##black_patch = mpatches.Patch(color='black', label='GridSearch')
#
#plt.legend(handles=[red_patch, blue_patch, green_patch])
#plt.plot(AUC_GA, color = 'blue', linewidth=1, linestyle='-', )
#plt.plot(AUC_PSO, color = 'red', linewidth=1, linestyle='-')
#plt.plot(tracking_Optuna_AUC, color = 'green', linewidth=1, linestyle='-')
##plt.plot(tracking_GS, color = 'black', linewidth=3, linestyle='-')
#plt.show()

# %%
#fig, ax = plt.subplots(2, 2, figsize=(25,15))
#
#ax[0, 0].set_title(f'Genetic Algorithm (GA) - Accuracy: {max(tracking_GA_AUC)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#ax[0, 1].set_title(f'Particle Swarm Optimization (PSO) - Accuracy: {max(tracking_PSO_AUC)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#
#ax[1, 0].set_title(f'Grid Search - Accuracy: {max(tracking_GS_AUC)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#ax[1, 1].set_title(f'Optuna - Accuracy: {max(tracking_Optuna_AUC)}', fontdict={'fontsize': 20, 'fontweight': 'medium'})
#
#ax[0,0].plot(tracking_GA_AUC, color = 'blue', linewidth=3, linestyle='-', )
#ax[0,1].plot(tracking_PSO_AUC, color = 'red', linewidth=3, linestyle='-')
#
#ax[1,0].plot(tracking_GS_AUC, color = 'black', linewidth=3, linestyle='-')
#ax[1,1].plot(tracking_Optuna_AUC, color = 'green', linewidth=3, linestyle='-')
#

# %%
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def random_search_int_range(min_l, max_l):
    return (np.arange(randint.ppf(0.01, min_l, max_l),randint.ppf(0.99, min_l, max_l))).astype(int)

def run_random_search_accuracy(iterations = 1024):
    param_distribution = {
    'n_estimators' : random_search_int_range(n_estimators_min, n_estimators_max),
    'learning_rate' : uniform(learning_rate_min, learning_rate_max),
    'subsample' : uniform(subsample_min, subsample_max - 1),
    'colsample_bytree' : uniform(colsample_bytree_min, colsample_bytree_max  - 1),
    'gamma' : uniform(gamma_min, gamma_max),
    'max_depth' : random_search_int_range(max_depth_min, max_depth_max),
    'min_child_weight' : random_search_int_range(min_child_weight_min, min_child_weight_max),
    'reg_alpha' : uniform(reg_alpha_min, reg_alpha_max),
    'reg_lambda' : uniform(reg_lambda_min, reg_lambda_max),
    'scale_pos_weight' : random_search_int_range(scale_pos_weight_min, scale_pos_weight_max),
    'base_score' : uniform(base_score_min, base_score_max),
    'n_jobs' :  [-1]
    }
    





    kfold = KFold(n_splits = 3, shuffle = True)

    random_search_accuracy = RandomizedSearchCV(xgb.XGBClassifier(), param_distribution, n_iter = iterations, n_jobs = -1, cv=kfold, verbose=False)
    random_search_accuracy.fit(x_heart, y_heart)

    return random_search_accuracy.best_score_

# %%
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def random_search_int_range(min_l, max_l):
    return (np.arange(randint.ppf(0.01, min_l, max_l),randint.ppf(0.99, min_l, max_l))).astype(int)


def run_random_search_f1(iterations = 1024):
    param_distribution = {
    'n_estimators' : random_search_int_range(n_estimators_min, n_estimators_max),
    'learning_rate' : uniform(learning_rate_min, learning_rate_max),
    'subsample' : uniform(subsample_min, subsample_max - 1),
    'colsample_bytree' : uniform(colsample_bytree_min, colsample_bytree_max  - 1),
    'gamma' : uniform(gamma_min, gamma_max),
    'max_depth' : random_search_int_range(max_depth_min, max_depth_max),
    'min_child_weight' : random_search_int_range(min_child_weight_min, min_child_weight_max),
    'reg_alpha' : uniform(reg_alpha_min, reg_alpha_max),
    'reg_lambda' : uniform(reg_lambda_min, reg_lambda_max),
    'scale_pos_weight' : random_search_int_range(scale_pos_weight_min, scale_pos_weight_max),
    'base_score' : uniform(base_score_min, base_score_max),
    'n_jobs' :  [-1]
    }
    




    kfold = KFold(n_splits = 3, shuffle = True)

    random_search_accuracy = RandomizedSearchCV(xgb.XGBClassifier(), param_distribution, n_iter = iterations, n_jobs = -1, cv=kfold, verbose=False, scoring='f1')
    random_search_accuracy.fit(x_heart, y_heart)

    return random_search_accuracy.best_score_


# %%
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def random_search_int_range(min_l, max_l):
    return (np.arange(randint.ppf(0.01, min_l, max_l),randint.ppf(0.99, min_l, max_l))).astype(int)

def run_random_search_auc(iterations = 1024):
    param_distribution = {
    'n_estimators' : random_search_int_range(n_estimators_min, n_estimators_max),
    'learning_rate' : uniform(learning_rate_min, learning_rate_max),
    'subsample' : uniform(subsample_min, subsample_max - 1),
    'colsample_bytree' : uniform(colsample_bytree_min, colsample_bytree_max  - 1),
    'gamma' : uniform(gamma_min, gamma_max),
    'max_depth' : random_search_int_range(max_depth_min, max_depth_max),
    'min_child_weight' : random_search_int_range(min_child_weight_min, min_child_weight_max),
    'reg_alpha' : uniform(reg_alpha_min, reg_alpha_max),
    'reg_lambda' : uniform(reg_lambda_min, reg_lambda_max),
    'scale_pos_weight' : random_search_int_range(scale_pos_weight_min, scale_pos_weight_max),
    'base_score' : uniform(base_score_min, base_score_max),
    'n_jobs' :  [-1]
    }
    
    




    kfold = KFold(n_splits = 3, shuffle = True)

    random_search_accuracy = RandomizedSearchCV(xgb.XGBClassifier(), param_distribution, n_iter = iterations, n_jobs = -1, cv=kfold, verbose=False, scoring='roc_auc')
    random_search_accuracy.fit(x_heart, y_heart)

    return random_search_accuracy.best_score_

# %% [markdown]
# # Gerando Dados Para Análise 

# %%
from tqdm import tqdm
import pandas as pd
from os import system

def full_run_optuna(iterations):
    filename = './XGBoost_heart_data.csv'

    try:
        XGBoost_data = pd.read_csv(filename)
    except:
        open(filename, "a")
        XGBoost_data = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'F1', 'AUC'])


    for i in tqdm(range(iterations)):      
        accuracy_optuna, _ = run_optuna_accuracy()
        f1_optuna, _ = run_optuna_f1()
        auc_optuna, _ = run_optuna_auc()
        temp = pd.DataFrame({'Algorithm' : ['Optuna'], 
                            'Accuracy' :[accuracy_optuna], 
                            'F1' : [f1_optuna], 
                            'AUC' : [auc_optuna]})
        #XGBoost_data = XGBoost_data.append(temp, ignore_index = True)
        XGBoost_data = pd.concat([XGBoost_data, temp], ignore_index=True)
        XGBoost_data[['Algorithm', 'Accuracy', 'F1', 'AUC']].to_csv(filename)

# %%
from tqdm import tqdm
import pandas as pd
from os import system

def full_run_ga(iterations):
    filename = './XGBoost_heart_data.csv'

    try:
        XGBoost_data = pd.read_csv(filename)
    except:
        open(filename, "a")
        XGBoost_data = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'F1', 'AUC'])


    for i in tqdm(range(iterations)):

        accuracy_ga, _, _ = run_accuracy_ga()
        f1_ga, _, _ = run_f1_ga()
        auc_ga, _, _ = run_auc_ga()
        temp = pd.DataFrame({'Algorithm' : ['GA'], 
                            'Accuracy' :[-accuracy_ga[0]], 
                            'F1' : [-f1_ga[0]], 
                            'AUC' : [-auc_ga[0]]})
        #XGBoost_data = XGBoost_data.append(temp, ignore_index = True)
        XGBoost_data = pd.concat([XGBoost_data, temp], ignore_index=True)
        XGBoost_data[['Algorithm', 'Accuracy', 'F1', 'AUC']].to_csv(filename)
        

# %%
from tqdm import tqdm
import pandas as pd
from os import system

def full_run_optuna(iterations):
    filename = './XGBoost_heart_data.csv'

    try:
        XGBoost_data = pd.read_csv(filename)
    except:
        open(filename, "a")
        XGBoost_data = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'F1', 'AUC'])


    for i in tqdm(range(iterations)):      
        accuracy_optuna, _ = run_optuna_accuracy()
        f1_optuna, _ = run_optuna_f1()
        auc_optuna, _ = run_optuna_auc()
        temp = pd.DataFrame({'Algorithm' : ['Optuna'], 
                            'Accuracy' :[accuracy_optuna], 
                            'F1' : [f1_optuna], 
                            'AUC' : [auc_optuna]})
        #XGBoost_data = XGBoost_data.append(temp, ignore_index = True)
        XGBoost_data = pd.concat([XGBoost_data, temp], ignore_index=True)
        XGBoost_data[['Algorithm', 'Accuracy', 'F1', 'AUC']].to_csv(filename)

# %%
from tqdm import tqdm
import pandas as pd
from os import system

def full_run_pso(iterations):
    filename = './XGBoost_heart_data.csv'

    try:
        XGBoost_data = pd.read_csv(filename)
    except:
        open(filename, "a")
        XGBoost_data = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'F1', 'AUC'])


    for i in tqdm(range(iterations)):
        accuracy_pso = run_accuracy_pso()
        f1_pso = run_f1_pso()
        auc_pso = run_auc_pso()
        temp = pd.DataFrame({'Algorithm' : ['PSO'], 
                            'Accuracy' :[accuracy_pso], 
                            'F1' : [f1_pso], 
                            'AUC' : [auc_pso]})
        #XGBoost_data = XGBoost_data.append(temp, ignore_index = True)
        XGBoost_data = pd.concat([XGBoost_data, temp], ignore_index=True)
        XGBoost_data[['Algorithm', 'Accuracy', 'F1', 'AUC']].to_csv(filename)
        

# %%
def full_run_random_searh(iterations):
    filename = './XGBoost_heart_data.csv'

    try:
        XGBoost_data = pd.read_csv(filename)
    except:
        open(filename, "a")
        XGBoost_data = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'F1', 'AUC'])


    for i in tqdm(range(iterations)):

        accuracy_rs = run_random_search_accuracy()
        f1_rs = run_random_search_f1()
        auc_rs = run_random_search_auc()
        temp = pd.DataFrame({'Algorithm' : ['Random Search'], 
                            'Accuracy' :[accuracy_rs], 
                            'F1' : [f1_rs], 
                            'AUC' : [auc_rs]})
        #XGBoost_data = XGBoost_data.append(temp, ignore_index = True)
        XGBoost_data = pd.concat([XGBoost_data, temp], ignore_index=True)
        XGBoost_data[['Algorithm', 'Accuracy', 'F1', 'AUC']].to_csv(filename)


full_run_optuna(30)