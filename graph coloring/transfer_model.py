import itertools as it
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from dataset_pytorch import SAT3Dataset
from model import GNN
import warnings
warnings.filterwarnings('ignore')

MAX_NUMBER_OF_EPOCHS = 51
EARLY_STOPPING_COUNTER = 15

# set seed so that the train-valid sets are always the same
torch.manual_seed(15)
torch.cuda.manual_seed(15)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device is: {device}')


def tune_parameters(pos_weight, model_name, best_pre_trained):

    # parameters that have to do with the training
    hyperparameters_options_algo = {
        "batch_size": [16, 32],
        "learning_rate": [0.001, 0.005, 0.01],
        "weight_decay": [0.00001, 0.0001, 0.001],
        "pos_weight": [pos_weight]
    }

    best_pre_trained_parameters = best_pre_trained

    # Now, tune parameters that correspond to the algo and not the GNN 's architecture
    # we keep the tuned parameters from before
    algo_hyperparameters = {"batch_size": 0, "learning_rate": 0, "weight_decay": 0, "pos_weight": 0,
                            "model_embedding_size": best_pre_trained_parameters["model_embedding_size"],
                            "model_attention_heads": best_pre_trained_parameters["model_attention_heads"],
                            "model_layers": best_pre_trained_parameters["model_layers"],
                            "model_dropout_rate": best_pre_trained_parameters["model_dropout_rate"],
                            "model_dense_neurons": best_pre_trained_parameters["model_dense_neurons"]}

    # get all possible combination of the algo 's parameters
    algo_parameter_combinations = it.product(*(hyperparameters_options_algo[param] for param in
                                               hyperparameters_options_algo))
    counter = 0
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
        validation_loss = re_training(params=algo_hyperparameters,
                                      best_pre_trained_params=best_pre_trained_parameters, model_name=model_name)
        # choose these parameters if they give us a smaller validation set-loss
        if validation_loss < best_validation_loss:
            print('New best parameters found!\n')
            best_validation_loss = validation_loss
            best_parameters = algo_hyperparameters.copy()

        counter += 1

    return best_parameters


def train_one_epoch(model, train_loader, optimizer, criterion):
    running_loss = 0.0
    step = 0

    torch.manual_seed(15)
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        # make prediction
        prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        # calculate loss
        loss = criterion(torch.squeeze(prediction), batch.y.float())
        # calculate gradient
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

    return running_loss / step


def evaluation(model, test_loader, criterion):
    running_loss = 0.0
    step = 0

    torch.manual_seed(15)
    for batch in test_loader:
        batch.to(device)
        # make prediction
        prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        # calculate loss
        loss = criterion(torch.squeeze(prediction), batch.y.float())

        running_loss += loss.item()
        step += 1

    return running_loss / step


def re_training(params, best_pre_trained_params, model_name, train_set="store.h5"):
    # loading the dataset
    print("Dataset loading...")
    dataset = SAT3Dataset(root="./", filename=train_set)

    # we have already kept a different test set, so split into train and validation 80% - 20%
    train_set_size = np.ceil(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size

    torch.manual_seed(15)
    train_dataset, valid_dataset = \
        torch.utils.data.random_split(dataset, [int(train_set_size), int(valid_set_size)])

    # no shuffling, as it is already shuffled
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"])
    valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"])

    print("Dataset loading completed\n")

    best_pre_trained_params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

    # Load the pre-trained GNN model
    print("Model loading...")
    model_params = {k: v for k, v in best_pre_trained_params.items() if k.startswith("model_")}
    best_model = GNN(feature_size=dataset[0].x.shape[1], model_params=model_params)
    best_model.load_state_dict(torch.load(model_name, map_location="cuda:0"))

    # Freeze the layers
    for param in best_model.parameters():
        param.requires_grad = False

    # Modify the last linear layer that will be re-trained
    #number_features1 = best_model.linear1.in_features
    #number_features2 = best_model.linear2.in_features
    number_features3 = best_model.linear3.in_features

    #best_model.linear1 = torch.nn.Linear(number_features1, number_features2)
    #best_model.linear2 = torch.nn.Linear(number_features2, number_features3)
    best_model.linear3 = torch.nn.Linear(number_features3, 1)

    best_model.to(device)

    print("Model loading completed\n")

    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)

    # define a loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    optimizer = torch.optim.Adam(best_model.parameters(), lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"], amsgrad=False)

    # no parameter optimizing for 'scheduler gamma' as it multiplies with weight decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # START TRAINING

    # initialize some parameters
    loss_diff = 1.0             # train and validation loss difference, to avoid overfitting
    final_valid_loss = 1000     # validation loss
    final_train_loss = 1000     # training loss
    early_stopping_counter = 0  # counter for early stopping

    for epoch in range(MAX_NUMBER_OF_EPOCHS):

        print(f'EPOCH | {epoch}')

        if early_stopping_counter < EARLY_STOPPING_COUNTER:
            # perform one training epoch
            best_model.train()
            training_loss = train_one_epoch(best_model, train_loader, optimizer, criterion)
            print(f"Training Loss   : {training_loss:.4f}")

            # compute validation set loss
            best_model.eval()
            validation_loss = evaluation(best_model, valid_loader, criterion)
            print(f"Validation Loss : {validation_loss:.4f}\n")

            # check for early stopping if model is yet to finish its training
            difference = abs(float(validation_loss) - float(training_loss))
            if difference < loss_diff:
                loss_diff = difference
                final_valid_loss = validation_loss
                final_train_loss = training_loss
                # if still some progress can be made -> save the currently best model
                torch.save(best_model.state_dict(), model_name[:-4]+"_c"+model_name[-4:])

                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            scheduler.step()
        else:
            difference = abs(float(final_valid_loss) - float(final_train_loss))
            print(f"Early stopping activated, with training and validation loss difference: {difference:.4f}")

            return final_valid_loss

    print(f"Finishing training with best training loss: {final_train_loss:.4f} and best "
          f"validation loss: {final_valid_loss:.4f}")

    return final_valid_loss

