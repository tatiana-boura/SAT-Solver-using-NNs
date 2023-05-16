import sys, os, shutil, json
sys.path.insert(0, '../classification/GNN')

from train import testing
from data_loader_c import dataset_processing


def delete_folder_contents(folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


# adding Folder_2 to the system path
gnn_path = "../classification/GNN/"

delete_folder_contents([gnn_path+"raw", gnn_path+"processed"])


dataset_processing()

# Access the best parameters in order to train final model
with open(gnn_path+'best_parameters_same_sets.txt') as f:
    data = f.read()

best_parameters_loaded = json.loads(data)
print('\nResults on the test set:\n')

testing(params=best_parameters_loaded, model_name=gnn_path+'final_model_same_sets.pth',
        test_set="C:/Users/tatbo/PycharmProjects/MSc-in-AI-Demokritos-Deep-Learning-Course/graph coloring/store.h5")