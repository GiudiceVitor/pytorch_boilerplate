# PYTORCH BOILERPLATE

This is an open-source boilerplate for PyTorch projects. It organizes and optimizes a typical PyTorch project, dividing each component into its designed file. It creates a plug-n-play easy to use project structure focusing on requiring minimal changing in the overall structure of the main file ([project.py](source/project.py)).

Please notice that this is focused on the most standard use cases of a machine learning project (classification, regression, segmentation, etc), and will require further adaptation of the training loop for more diverse and complex projects.

## Project Structure

The project is divided into 3 folders:

- **data**: here lies the data of the project.
- **source**: here lies the source code. It is divided as follows:
    - [dataset.py](source/dataset.py): contains the definition of the dataset class.
    - [neural_network.py](source/neural_network.py): contains the PyTorch model.
    - [losses.py](source/losses.py): contains the losses and metrics used to train and evaluate the model.
    - [project.py](source/project.py): is the main file of the source code. Contains the training loop, evaluation and prediction subroutines, model compilation, etc.
    - [callbacks.py](source/callbacks.py): contains the callbacks of the project. This is the main form of altering the training loop. Here should lie classes for logging, scheduling the learning rate, changing parameteres mid-training, etc.
- **notebooks**: here lies the experimental notebooks. Here you can find [test.ipynb](notebooks/test.ipynb), a notebook with the basic testing of the project on the MNIST dataset. This dataset can be found [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
 