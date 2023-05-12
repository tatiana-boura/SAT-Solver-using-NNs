import os, shutil
import json

from tuning import tune_parameters
from data_loader import dataset_processing
from train import training, testing


def delete_folder_contents(folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


# make the dataset
'''
delete_folder_contents(["./raw", "./processed"])

pos_weight = dataset_processing(separate_test=False)

# tune the model
best_parameters = tune_parameters(pos_weight=pos_weight)

# best parameters are selected. View test-set metrics
print(f'Best hyperparameters were: {best_parameters}')
# store best parameters into a file
with open('best_parameters_same_sets.txt', 'w') as f:
    f.write(json.dumps(best_parameters))

f.close()
'''
# now access the best parameters in order to train final model

# reading the data from the file
with open('best_parameters_same_sets.txt') as f:
    data = f.read()

best_parameters_loaded = json.loads(data)
print(best_parameters_loaded)
'''
print('\nNow training with the best parameters\n')
training(best_parameters_loaded, make_err_logs=True)
'''
print('\nResults on the test set:\n')
testing(params=best_parameters_loaded)
'''
delete_folder_contents(["./raw", "./processed"])

"""start new section: train with different set"""
print("\n\nTRY SEPARATING TEST SET (BIGGER PROBLEM) FROM THE BEGINNING\n\n")


# make the dataset
pos_weight = dataset_processing(separate_test=True)

# tune the model
best_parameters = tune_parameters(pos_weight=pos_weight)

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
training(best_parameters_loaded, make_err_logs=True)

print('\nResults on the test set:\n')
testing(params=best_parameters_loaded)

'''