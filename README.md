# Predicting Satisfiability in SAT3 and Graph Coloring Problems using Graph Transformers Networks (GTNs) and Long Short-Term Memory Networks (LSTMs)
#### MSc-in-AI-Demokritos-Deep-Learning-Course  
#### Author:  Tatiana Boura MTN2210
------------------------------------------------

## Contents of this repository: 
- [Report.pdf] - The report of the project.
- [Presentation.pdf] - The presentation of the project.
- [requirements.txt] - Requirements.
- [datasets] - Folder that includes the dataset for the *SAT3 Problem* and the dataset for the *Graph-Coloring Problem*.
- [classification/GNN] - Folder that includes all prerequisites for the GTN training-tuning-evaluation. More specifically,

    * ``data_loader.py``, data loader that processes raw .cnf data for the SAT3 problem,
    * ``dataset_pytorch.py``, data loader that translates the processed data to the *torch.geometric* format,
    * ``model.py``, where the *GTN model architecture* is located,
    * ``tuning.py``, where functions for *hyperparameter tuning* are located,
    * ``train.py``, where functions for *training*, *testing* and *logging* are located, 
    * ``best_parameters_same_sets.txt`` & ``best_parameters_diff_test.txt``, where best parameters are stored after tuning (for same distribution sets and different distribution sets, respectively),
    * ``final_model_same_sets.pth``,``final_model_diff_sets.pth`` & ``final_model_same_sets_c.pth``, where best models are stored after tuning (for same distribution sets, different distribution sets and graph coloring, respectively),
    * folder **>plots**, where logging plots are located,
    * ``demo.py``, demo that showcases the performance of best model (*when trained and tested on data from same distribution*) on the unseen testing example ``demo.cnf``,
    * [GNN/classification.ipynb], a Python notebook where the model is **trained, tuned and evaluated** (on two different evaluation sets).

- [classification/LSTM] - Folder that includes all prerequisites for the LSTM training-tuning-evaluation. More specifically,

    * ``data_loader.py``, data loader that processes raw .cnf data for the SAT3 problem,
    * ``dataset_pytorch.py``, data loader that translates the processed data to the *torch.utils.data* format,
    * ``model.py``, where the *LSTM model architecture* is located,
    * ``tuning.py``, where functions for *hyperparameter tuning* are located,
    * ``train.py``, where functions for *training*, *testing* and *logging* are located, 
    * ``best_parameters_same_sets.txt``, where best parameters are stored after tuning,
    * ``final_model.pth``, where best model is stored after tuning,
    * folder **>plots**, where logging plots are located,
    * [LSTM/classification.ipynb], a Python notebook where the model is **trained, tuned and evaluated**.

- [graph coloring] - Folder that includes all prerequisites for *transfer learning* the best GTN model for the SAT3 problem to the Graph coloring problem. More specifically,

    * ``data_loader_c.py``, data loader that processes raw .cnf data for the 3-colorable problem,
    * ``transfer_model.py``, where functions for *re-training* and *hyperparameter tuning* are located,
    * ``best_parameters_same_sets.txt``, where best parameters are stored after tuning the learning parameters of the transferred model,
    * folder **>plots**, where logging plots are located,
    * [graph_coloring/test_model.ipynb], a Python notebook where the model is **re-trained and evaluated**.


### Process
In order to run the whole process you should execute the notebooks,
>    1. [classification/GNN/classification.ipynb]
>    2. [classification/LSTM/classification.ipynb]
>    3. [graph_coloring/test_model.ipynb]

with the given order. 
    
However, every notebook **can also be executed separately**.

Otherwise, you could just run the [demo]. Note that the demo is only for the best-performing model: the one trained and tested on data from the same distribution of SAT3 problem instances.



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[requirements.txt]:   
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/requirements.txt>
[Presentation.pdf]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/Presentation.pdf>
[Report.pdf]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/Report.pdf>
[datasets]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/tree/main/dataset>
[classification/GNN]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/tree/main/classification/GNN>
[classification/LSTM]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/tree/main/classification/LSTM>
[classification/GNN/classification.ipynb]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/classification/GNN/classification.ipynb>
[classification/LSTM/classification.ipynb]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/classification/LSTM/classification.ipynb>
[GNN/classification.ipynb]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/classification/GNN/classification.ipynb>
[LSTM/classification.ipynb]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/classification/LSTM/classification.ipynb>
[graph coloring]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/tree/main/graph%20coloring>
[graph_coloring/test_model.ipynb]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/graph%20coloring/test_model.ipynb>
[demo]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Deep-Learning-Course/blob/main/classification/GNN/demo.py>


