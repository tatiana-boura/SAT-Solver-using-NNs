import json

from tuning import tune_parameters
from data_loader import dataset_processing
from train import training, testing

# make the dataset
pos_weight = dataset_processing()

# tune the model
best_parameters = tune_parameters(pos_weight=pos_weight)

# best parameters are selected. View test-set metrics
print(f'Best hyperparameters were: {best_parameters}')
# store best parameters into a file
with open('best_parameters.txt', 'w') as f:
    f.write(json.dumps(best_parameters))

f.close()

# now access the best parameters in order to train final model

# reading the data from the file
with open('best_parameters.txt') as f:
    data = f.read()

best_parameters_loaded = json.loads(data)
print(best_parameters_loaded)

print('\nNow training with the best parameters\n')
training(params=best_parameters_loaded)

print('\nResults on the test set:\n')
testing(params=best_parameters_loaded)
