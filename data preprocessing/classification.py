import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset_pytorch import SAT3Dataset
from model import GNN

# set seed so that the train-test-valid sets are always the same
torch.manual_seed(15)

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def save_model(epoch, model, optimizer, criterion):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, './final_model.pth')


def metrics(y_pred, y, epoch):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y)}")
    # save confusion matrix as plot
    cm = confusion_matrix(y_pred, y)
    classes = ["UNSATISFIABLE", "SATISFIABLE"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'./plots/cm_{epoch}.png')

    # compute metrics
    print(f"F1 Score  : {f1_score(y, y_pred):.4f}")
    print(f"Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f}")
    # auc score
    roc = roc_auc_score(y, y_pred)
    print(f"ROC AUC   : {roc:.4f}")


def train_one_epoch(model, train_loader, optimizer, criterion):
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        optimizer.zero_grad()
        # make prediction
        prediction = model(batch.x.float(), batch.edge_attr.float(),batch.edge_index,batch.batch)
        # calculate loss
        loss = criterion(torch.squeeze(prediction), batch.y.float())
        # calculate gradient
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

    return running_loss / step


def evaluation(epoch, model, test_loader, criterion, test_set=False, print_metrics=False):
    predictions = []
    labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)
        # make prediction
        prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        # calculate loss
        loss = criterion(torch.squeeze(prediction), batch.y.float())

        running_loss += loss.item()
        step += 1

        if print_metrics:
            predictions.append(np.rint(torch.sigmoid(prediction).cpu().detach().numpy()))
            labels.append(batch.y.cpu().detach().numpy())

    if print_metrics:
        predictions = np.concatenate(predictions).ravel()
        labels = np.concatenate(labels).ravel()

        metrics(predictions, labels, epoch)

    return running_loss / step


def training(params, final_training=False):
    # loading the dataset
    print("Dataset loading...")
    dataset = SAT3Dataset(root="./", filename="store.h5")

    # split into train, validation and test 60 - 20 - 20
    train_set_size = np.ceil(dataset.len() * 0.6)
    valid_set_size = np.ceil(dataset.len() * 0.2)
    test_set_size = dataset.len() - (train_set_size + valid_set_size)

    train_dataset, valid_dataset, test_dataset = \
        torch.utils.data.random_split(dataset, [int(train_set_size), int(valid_set_size), int(test_set_size)])

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])  # no need to shuffle the test set
    print("Dataset loading completed")

    params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

    # load the GNN model
    print("Model loading...")
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
    model = model.to(device)
    print("Model loading completed")

    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)

    # define a loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"],
                                 amsgrad=False)
    '''
    # define an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"],
                                                    momentum=params["sgd_momentum"],
                                                    weight_decay=params["weight_decay"])
    '''
    # no parameter optimizing for 'scheduler gamma' as it multiplies with weight decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # START TRAINING

    # initialize some parameters
    loss_diff = 1.0             # train and validation loss difference, to avoid overfitting
    final_valid_loss = 1000     # validation loss
    final_train_loss = 1000     # training loss
    max_number_of_epochs = 50   # number of epochs for training
    early_stopping_counter = 0  # counter for early stopping

    for epoch in range(max_number_of_epochs):

        print(f'EPOCH | {epoch}')

        if early_stopping_counter < 8:
            # perform one training epoch
            model.train()
            training_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            print(f"Training Loss   : {training_loss:.4f}")

            # compute validation set loss
            model.eval()
            validation_loss = evaluation(epoch, model, valid_loader, criterion)
            print(f"Validation Loss : {validation_loss:.4f}\n")

            # check for early stopping
            difference = abs(float(validation_loss) - float(training_loss))
            if difference < loss_diff:
                loss_diff = difference
                final_valid_loss = validation_loss
                final_train_loss = training_loss
                # if still some progress can be made -> save the currently best model
                if final_training:
                    torch.save(model.state_dict(), './final_model.pth')

                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            scheduler.step()
        else:
            difference = abs(float(final_valid_loss) - float(final_train_loss))
            print(f"Early stopping activated, with training and validation loss difference: {difference:.4f}")

            if final_training:
                print("\nTest set metrics:")
                test_loss = evaluation(epoch-1, model, test_loader, criterion, test_set=True, print_metrics=True)
                print(f"Test Loss {test_loss}")
                # see test set's metrics in the best (not overfitted) model
                best_model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
                best_model.load_state_dict(torch.load('./final_model.pth', map_location="cuda:0"))
                best_model.to(device)
                best_model.eval()

                print("\nTest set metrics:")
                test_loss = evaluation(epoch - 1, best_model, test_loader, criterion, test_set=True, print_metrics=True)
                print(f"Test Loss 2 {test_loss}")

            return final_valid_loss

    print(f"Finishing training with best training loss: {final_train_loss:.4f} and best "
          f"validation loss: {final_valid_loss:.4f}")

    if final_training:
        print("\nTest set metrics:")
        test_loss = evaluation(epoch - 1, model, test_loader, criterion, test_set=True, print_metrics=True)
        print(f"Test Loss at last model : {test_loss}")

        best_model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
        best_model.load_state_dict(torch.load('./final_model.pth', map_location="cuda:0"))
        best_model.to(device)
        best_model.eval()

        print("\nTest set metrics:")
        test_loss = evaluation(epoch - 1, best_model, test_loader, criterion, test_set=True, print_metrics=True)
        print(f"Test Loss at final model : {test_loss}")

    return final_valid_loss


