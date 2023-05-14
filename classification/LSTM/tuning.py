from train import training
import itertools as it


def tune_parameters(pos_weight):

    # parameters that have to do with the LSTM
    hyperparameters_options_model = {
        "model_hidden_units": [8, 16, 32, 64],
        "model_num_layers": [1, 3, 5],
        "model_dropout": [0.0, 0.3, 0.5, 0.8],
    }

    # parameters that have to do with the training
    hyperparameters_options_algo = {
        "batch_size": [16, 32, 64],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "weight_decay": [0.00001, 0.0001, 0.001],
        "pos_weight": [pos_weight]
    }

    model_hyperparameters = {"batch_size": 16, "learning_rate": 0.05, "weight_decay": 0.001, "pos_weight": pos_weight,
                             "model_hidden_units": 16, "model_num_layers": 1, "model_dropout": 0.7}

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
        model_hyperparameters["model_dropout"] = parameters[2]
        # try training with these parameters
        print(f'\nTest number {counter} | Start testing new parameter-combination...\n')
        validation_loss = training(params=model_hyperparameters)
        # choose these parameters if they give us a smaller validation set-loss
        if validation_loss < best_validation_loss:
            print('New best parameters found!\n')
            best_validation_loss = validation_loss
            best_model_parameters = model_hyperparameters.copy()

        counter += 1

    # Now, tune parameters that correspond to the algo and not the LSTM's architecture
    # we keep the tuned parameters from before
    algo_hyperparameters = {"batch_size": 0, "learning_rate": 0, "weight_decay": 0, "pos_weight": 0,
                            "model_hidden_units": best_model_parameters["model_hidden_units"],
                            "model_num_layers": best_model_parameters["model_num_layers"],
                            "model_dropout": best_model_parameters["model_dropout"]}

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
