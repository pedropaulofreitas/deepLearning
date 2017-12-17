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
				'activation_dropout_nlayers': hp.choice('activation_dropout_type',
										[{
											# 'activation': 'selu',
											# 'dropout_rate': hp.choice('dr', (0.0, 0.05, 0.1, 0.15)),
											# 'n_layers': hp.choice('nl', (1, 2, 4, 8)),
										# },
										# {
											'activation': 'tanh',
											## 'dropout_rate': hp.choice('dr', (0.0, 0.25, 0.5, 0.75)),
											'n_layers': hp.choice('nl', (1, 2, 3)),
										}]),
				'hidden_layers': hp.quniform('ls', 1, 200, 1 ),
				'batch_size': hp.choice('bs', ( 32, 64 )),
				# 'learning_rate': hp.loguniform('lr', -10, -1.5),
				'weight_decay': hp.loguniform('reg', -14, -3)
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
		pprint({ k: v for k, v in params.items() if not k.startswith('activation_dropout_nlayers')})
		pprint({ k: v for k, v in params['activation_dropout_nlayers'].items()})
		print()

	def try_params(self, n_iterations, params):

		print("iterations:", n_iterations)
		self.print_params(params)

		return self.train_fn(n_iterations,params)
