import json

from tuning import tune_parameters
from data_loader import dataset_processing
from train import training, testing

'''
separate_test = False

# make the dataset
pos_weight = dataset_processing(separate_test=separate_test)

# tune the model
best_parameters = tune_parameters(pos_weight=pos_weight, separate_test=separate_test)

# best parameters are selected. View test-set metrics
print(f'Best hyperparameters were: {best_parameters}')
# store best parameters into a file
with open('best_parameters_same_sets.txt', 'w') as f:
    f.write(json.dumps(best_parameters))

f.close()

# now access the best parameters in order to train final model

# reading the data from the file
with open('best_parameters_same_sets.txt') as f:
    data = f.read()

best_parameters_loaded = json.loads(data)
print(best_parameters_loaded)

print('\nNow training with the best parameters\n')
# training(best_parameters_loaded, make_err_logs=True)
training(params=best_parameters_loaded, separate_test=separate_test)

print('\nResults on the test set:\n')
testing(params=best_parameters_loaded, separate_test=separate_test)

'''
"""start new section: train with different set"""
print("\n\nTRY SEPARATING TEST SET (BIGGER PROBLEM) FROM THE BEGINNING\n\n")

separate_test = True

# make the dataset
pos_weight = dataset_processing(separate_test=separate_test)

# tune the model
best_parameters = tune_parameters(pos_weight=pos_weight, separate_test=separate_test)

# best parameters are selected. View test-set metrics
print(f'Best hyperparameters were: {best_parameters}')
# store best parameters into a file
with open('best_parameters_diff_test.txt', 'w') as f:
    f.write(json.dumps(best_parameters))

f.close()

# now access the best parameters in order to train final model

# reading the data from the file
with open('best_parameters_diff_test.txt') as f:
    data = f.read()

best_parameters_loaded = json.loads(data)
print(best_parameters_loaded)

print('\nNow training with the best parameters\n')
# training(best_parameters_loaded, make_err_logs=True)
training(params=best_parameters_loaded, separate_test=separate_test)

print('\nResults on the test set:\n')
testing(params=best_parameters_loaded, separate_test=separate_test)
