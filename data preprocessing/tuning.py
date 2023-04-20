import itertools as it
import json

from classification import training
from data_loader import dataset_processing

# make the dataset
pos_weight = dataset_processing()


hyperparameters_options_model = {
    "model_embedding_size": [64, 128],
    "model_attention_heads": [1],
    "model_layers": [2, 3],
    "model_dropout_rate": [0.2, 0.5],
    "model_dense_neurons": [128, 256]
}

hyperparameters_options_algo = {
    "batch_size": [32, 64],   # with 128, time error
    "learning_rate": [0.001, 0.01, 0.1],
    "weight_decay": [0.00001, 0.0001, 0.001],
    "pos_weight": [pos_weight]
}


model_hyperparameters = {"batch_size": 64, "learning_rate": 0.01, "weight_decay": 0.0001, "pos_weight": pos_weight,
                         "model_embedding_size": 0, "model_attention_heads": 0, "model_layers": 0,
                         "model_dropout_rate": 0, "model_dense_neurons": 0}

# get all possible combination of the model's parameters
model_parameter_combinations = it.product(*(hyperparameters_options_model[param] for param in
                                            hyperparameters_options_model))

best_validation_loss = 100000.0
best_model_parameters = model_hyperparameters.copy()

counter = 1
# at first, tune parameters that have to do with the model's architecture
for parameters in list(model_parameter_combinations):
    # make current parameters
    model_hyperparameters["model_embedding_size"] = parameters[0]
    model_hyperparameters["model_attention_heads"] = parameters[1]
    model_hyperparameters["model_layers"] = parameters[2]
    model_hyperparameters["model_dropout_rate"] = parameters[3]
    model_hyperparameters["model_dense_neurons"] = parameters[4]

    # try training with these parameters
    print(f'\nTest number {counter} | Start testing new parameter-combination...\n')
    validation_loss = training(model_hyperparameters)

    if validation_loss < best_validation_loss:
        print('New best parameters found!\n')
        best_validation_loss = validation_loss
        best_model_parameters = model_hyperparameters.copy()

    counter += 1

print(f'\nBest model hyperparameters were: {best_model_parameters}')

# now, tune parameters that correspond to the algo and not the GNN's architecture
# we keep the tuned parameters from before
algo_hyperparameters = {"batch_size": 0, "learning_rate": 0, "weight_decay": 0, "pos_weight": 0,
                        "model_embedding_size": best_model_parameters["model_embedding_size"],
                        "model_attention_heads": best_model_parameters["model_attention_heads"],
                        "model_layers": best_model_parameters["model_layers"],
                        "model_dropout_rate": best_model_parameters["model_dropout_rate"],
                        "model_dense_neurons": best_model_parameters["model_dense_neurons"]}

# get all possible combination of the algo's parameters
algo_parameter_combinations = it.product(*(hyperparameters_options_algo[param] for param in
                                           hyperparameters_options_algo))

best_validation_loss = 100000.0
best_parameters = algo_hyperparameters.copy()

for parameters in list(algo_parameter_combinations):
    # make current parameters
    algo_hyperparameters["batch_size"] = parameters[0]
    algo_hyperparameters["learning_rate"] = parameters[1]
    algo_hyperparameters["weight_decay"] = parameters[2]
    algo_hyperparameters["pos_weight"] = parameters[3]

    # try training with these parameters
    print(f'\nTest number {counter} | Start testing new parameter-combination...\n')
    validation_loss = training(algo_hyperparameters)

    if validation_loss < best_validation_loss:
        print('New best parameters found!\n')
        best_validation_loss = validation_loss
        best_parameters = algo_hyperparameters.copy()

    counter += 1

print(f'Best hyperparameters were: {best_parameters}')
print('\nNow training with the best parameters\n')
validation_loss = training(best_parameters, final_training=True)

# store best parameters into a file
with open('best_parameters.txt', 'w') as convert_file:
    convert_file.write(json.dumps(best_parameters))

