"function (and parameter space) definitions for hyperband"
"regression with multilayer perceptron"

# a dict with x_train, y_train, x_test, y_test
# from load_data_for_regression import data

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

from pprint import pprint

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

class DefOptim:

	def __init__(self, train_fn):
		self.space = {
				'n_estimators': hp.qloguniform("nest",1,7,1),
				'max_features':hp.choice('maxf',("log2", "auto")),
				'max_depth': hp.quniform('maxd',10,50,1),
				'min_samples_leaf':hp.choice('leaf',(1,2,3,4,5))
				}

		self.train_fn = train_fn

    	# handle floats which should be integers
	# works with flat params
	def handle_integers(self,params):
		new_params = {}
		for k, v in params.items():
			if type(v) == float and int(v) == v:
				new_params[k] = int(v)
			else:
				new_params[k] = v
		return new_params

	def get_params(self):
		params = sample(self.space)
		return self.handle_integers(params)

	def print_params(self,params):
		pprint({ k: v for k, v in params.items() })
		# pprint({ k: v for k, v in params['activation_dropout_nlayers'].items()})
		# print('batata')

	def try_params(self, n_iterations, params):
		print("iterations:", n_iterations)
		self.print_params(params)

		return self.train_fn(n_iterations,params)
