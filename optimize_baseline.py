from train_baseline import train

import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real, Integer, Categorical
# from skopt.utils import use_named_args
import argparse


import itertools
from joblib import Parallel, delayed
import random

class CustomGridSearch:
    def __init__(self, param_grid, scoring_function, n_jobs=1):
        self.param_grid = param_grid
        self.scoring_function = scoring_function
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []

    def fit(self, dataset_name):
        # Generate all possible combinations of parameters
        param_combinations = list(itertools.product(*(self.param_grid[param] for param in self.param_grid)))
        random.shuffle(param_combinations)
        print('Total combinations:', len(param_combinations))
        
        # Wrapper function to evaluate a single combination of parameters
        def evaluate_params(params):
            param_dict = {param: params[i] for i, param in enumerate(self.param_grid)}
            print(param_dict)
            score = self.scoring_function(dataset_name, **param_dict)
            return param_dict, score

        # Parallel processing of parameter combinations
        results = Parallel(n_jobs=self.n_jobs)(delayed(evaluate_params)(params) for params in param_combinations)
        
        # Process results to find the best parameters and score
        for param_dict, score in results:
            self.results_.append({'params': param_dict, 'score': score})
            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = param_dict
        
        return self

    def get_results(self):
        return self.results_

params  = {
    'epochs': [3000],
    'lr': [0.001, 0.01],
    'l2': [1e-4],
    'dropout': [0,0.5],
    'num_layers': [5],
    'hidden_dim': [16,32],
    'batch_size': [32,128],
}


def scoring_function(dataset_name, **params):
    score = train(dataset_name, params)['val_acc_mean']
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize_baseline.py')
    parser.add_argument('--dataset',  default='MUTAG', type=str, help='Dataset to use')
    parser.add_argument('--n_jobs',  default=10, type=int, help='Number of jobs')
    args = parser.parse_args()
    dataset_name = args.dataset
    n_jobs = args.n_jobs

    grid_search = CustomGridSearch(params, scoring_function, n_jobs=n_jobs)
    grid_search.fit(dataset_name)
    
    # Retrieve results
    results = grid_search.get_results()
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    # print("All Results:", results)

    