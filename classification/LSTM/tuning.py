from train import training
import itertools as it


def tune_parameters(pos_weight):

    hyperparameters_options_data = {
        "sequence_length": [1, 3, 5, 10]
    }

    # parameters that have to do with the GNN
    hyperparameters_options_model = {
        "model_hidden_units": [16, 32, 64],
        "model_num_layers": [1, 2, 3]
    }

    # parameters that have to do with the training
    hyperparameters_options_algo = {
        "batch_size": [32, 64, 128],
        "learning_rate": [0.001, 0.01, 0.1],
        "weight_decay": [0.00001, 0.0001, 0.001],
        "pos_weight": [pos_weight]
    }

    data_hyperparameters = {"batch_size": 64, "learning_rate": 0.01, "weight_decay": 0.0001, "pos_weight": pos_weight,
                            "sequence_length": 0, "model_hidden_units": 16, "model_num_layers": 1}

    # get all possible combination of the data's parameters
    data_parameter_combinations = it.product(*(hyperparameters_options_data[param] for param in
                                               hyperparameters_options_data))

    best_validation_loss = 100000.0
    best_data_parameters = data_hyperparameters.copy()

    counter = 1
    # at first, tune parameters that have to do with the data
    for parameters in list(data_parameter_combinations):
        # make current parameters
        data_hyperparameters["sequence_length"] = parameters[0]
        # try training with these parameters
        print(f'\nTest number {counter} | Start testing new parameter-combination...\n')
        validation_loss = training(params=data_hyperparameters)
        # choose these parameters if they give us a smaller validation set-loss
        if validation_loss < best_validation_loss:
            print('New best parameters found!\n')
            best_validation_loss = validation_loss
            best_data_parameters = data_hyperparameters.copy()

        counter += 1

    model_hyperparameters = {"batch_size": 64, "learning_rate": 0.01, "weight_decay": 0.0001, "pos_weight": pos_weight,
                             "sequence_length": best_data_parameters["sequence_length"], "model_hidden_units": 16,
                             "model_num_layers": 1}

    # get all possible combination of the model's parameters
    model_parameter_combinations = it.product(*(hyperparameters_options_model[param] for param in
                                                hyperparameters_options_model))

    best_validation_loss = 100000.0
    best_model_parameters = model_hyperparameters.copy()

    counter = 1
    # at first, tune parameters that have to do with the model's architecture
    for parameters in list(model_parameter_combinations):
        # make current parameters
        model_hyperparameters["model_hidden_units"] = parameters[0]
        model_hyperparameters["model_num_layers"] = parameters[1]
        # try training with these parameters
        print(f'\nTest number {counter} | Start testing new parameter-combination...\n')
        validation_loss = training(params=model_hyperparameters)
        # choose these parameters if they give us a smaller validation set-loss
        if validation_loss < best_validation_loss:
            print('New best parameters found!\n')
            best_validation_loss = validation_loss
            best_model_parameters = model_hyperparameters.copy()

        counter += 1

    # Now, tune parameters that correspond to the algo and not the GNN's architecture
    # we keep the tuned parameters from before
    algo_hyperparameters = {"batch_size": 0, "learning_rate": 0, "weight_decay": 0, "pos_weight": 0,
                            "sequence_length": best_data_parameters["sequence_length"],
                            "model_hidden_units": best_model_parameters["model_hidden_units"],
                            "model_num_layers": best_model_parameters["model_num_layers"]}

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
        validation_loss = training(params=algo_hyperparameters)
        # choose these parameters if they give us a smaller validation set-loss
        if validation_loss < best_validation_loss:
            print('New best parameters found!\n')
            best_validation_loss = validation_loss
            best_parameters = algo_hyperparameters.copy()

        counter += 1

    return best_parameters
