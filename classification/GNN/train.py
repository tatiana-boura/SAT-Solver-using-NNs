import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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


def plot_errors(errors, early_stopping):
    losses = list(map(list, zip(*errors)))
    train_loss = losses[0]
    valid_loss = losses[1]
    plt.plot([i+1 for i in range(MAX_NUMBER_OF_EPOCHS-1)], train_loss, color='r', label='Training loss')
    plt.plot([i + 1 for i in range(MAX_NUMBER_OF_EPOCHS-1)], valid_loss, color='g', label='Validation loss')
    plt.vlines(x=early_stopping, ymin=0.0, ymax=max(valid_loss), colors='purple', ls='--',
               label='early stopping activated')
    plt.vlines(x=early_stopping-EARLY_STOPPING_COUNTER, ymin=0.0, ymax=max(valid_loss), colors='magenta',
               label='considered model')

    plt.ylabel('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.title('Early stopping in selected model')

    plt.legend()
    plt.savefig('./plots/train_valid_error.png')
    plt.close()


def metrics(y_pred, y, y_proba=[]):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y)}")
    cm = confusion_matrix(y_pred, y)
    classes = ["UNSATISFIABLE", "SATISFIABLE"]
    ax = plt.axes()
    ax.set_title("Confusion Matrix")
    plt.close()
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g', ax=ax)
    cfm_plot.figure.savefig(f'./plots/cm.png')
    plt.close()

    # compute metrics
    print(f"F1 Score  : {f1_score(y, y_pred):.4f}")
    print(f"Accuracy  : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f}")
    # auc score

    one_class = False
    if sum(y) == len(y) or sum(y) == 0:
        one_class = True

    if not one_class:
        print(f"ROC AUC   : {roc_auc_score(y, y_pred):.4f}")

        # plot roc - auc curve
        RocCurveDisplay.from_predictions(y, y_pred)
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("FP Rate")
        plt.ylabel("TP Rate")
        plt.legend()
        plt.title("ROC-AUC curve")
        plt.savefig(f'./plots/roc_auc.png')
        plt.close()

    if len(y_proba) > 0:
        prec, recall, _ = precision_recall_curve(probas_pred=y_proba, y_true=y, pos_label=1.0)
        pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
        plt.close()
        fig, ax = plt.subplots()
        ax.set_title("Precision-Recall Curve")
        pr_display.plot(ax=ax)
        fig.figure.savefig(f'./plots/pr.png')
        plt.close()


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


def evaluation(model, test_loader, criterion, print_metrics=False):
    predictions = []
    predictions_proba = []
    labels = []
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

        if print_metrics:
            predictions.append(np.rint(torch.sigmoid(prediction).cpu().detach().numpy()))
            predictions_proba.append(torch.sigmoid(prediction).cpu().detach().numpy())
            labels.append(batch.y.cpu().detach().numpy())

    if print_metrics:
        predictions = np.concatenate(predictions).ravel()
        labels = np.concatenate(labels).ravel()
        predictions_proba = np.concatenate(predictions_proba).ravel()

        metrics(predictions, labels, predictions_proba)

    return running_loss / step


def training(params, model_name, make_err_logs=False):
    # loading the dataset
    print("Dataset loading...")
    dataset = SAT3Dataset(root="./", filename="store.h5")

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

    params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

    # load the GNN model
    print("Model loading...")
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
    model = model.to(device)
    print("Model loading completed\n")

    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)

    # define a loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"],
                                 amsgrad=False)

    # no parameter optimizing for 'scheduler gamma' as it multiplies with weight decay
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # START TRAINING

    # initialize some parameters
    loss_diff = 1.0             # train and validation loss difference, to avoid overfitting
    final_valid_loss = 1000     # validation loss
    final_train_loss = 1000     # training loss
    early_stopping_counter = 0  # counter for early stopping

    # the following are just for reporting reasons
    errors = []
    stopped = False
    early_stopping = MAX_NUMBER_OF_EPOCHS

    for epoch in range(MAX_NUMBER_OF_EPOCHS):

        print(f'EPOCH | {epoch}')

        if early_stopping_counter < EARLY_STOPPING_COUNTER:
            # perform one training epoch
            model.train()
            training_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            print(f"Training Loss   : {training_loss:.4f}")

            # compute validation set loss
            model.eval()
            validation_loss = evaluation(model, valid_loader, criterion)
            print(f"Validation Loss : {validation_loss:.4f}\n")

            errors += [(training_loss, validation_loss)]

            # check for early stopping if model is yet to finish its training
            if not stopped:
                difference = abs(float(validation_loss) - float(training_loss))
                if difference < loss_diff:
                    loss_diff = difference
                    final_valid_loss = validation_loss
                    final_train_loss = training_loss
                    # if still some progress can be made -> save the currently best model
                    torch.save(model.state_dict(), model_name)

                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            difference = abs(float(final_valid_loss) - float(final_train_loss))
            print(f"Early stopping activated, with training and validation loss difference: {difference:.4f}")

            if make_err_logs:
                print("\nTraining has stopped, now continuing with logging\n")
                stopped = True
                early_stopping_counter = 0
                early_stopping = epoch
            else:
                return final_valid_loss

    print(f"Finishing training with best training loss: {final_train_loss:.4f} and best "
          f"validation loss: {final_valid_loss:.4f}")

    if make_err_logs:
        plot_errors(errors, early_stopping)

    return final_valid_loss


def testing(params, model_name, test_set="store_test.h5"):
    # loading the dataset
    print("Dataset loading...")

    # dataset is different, just load it
    dataset = SAT3Dataset(root="./", filename=test_set, test=True)
    test_loader = DataLoader(dataset, batch_size=params["batch_size"])  # no need to shuffle the test set

    print("Dataset loading completed\n")

    ### ADDED LATER: CHAECK
    params["model_edge_dim"] = dataset[0].edge_attr.shape[1]

    # see test set's metrics in the best (not overfitted) model
    print("Model loading...")
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    best_model = GNN(feature_size=dataset[0].x.shape[1], model_params=model_params)
    best_model.load_state_dict(torch.load(model_name, map_location="cuda:0"))
    best_model.to(device)
    best_model.eval()
    print("Model loading completed\n")

    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)

    # define a loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    print("\nTest set metrics:")
    test_loss = evaluation(best_model, test_loader, criterion, print_metrics=True)
    print(f"Test Loss : {test_loss}")
