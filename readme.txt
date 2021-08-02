This folder contains the code associated to the study: "Deep learning retrofittting and risk assessment of road networks"
by Rodrigo Silva-Lopez, Jack W. Baker and Alan Poulos

The paper presents two main topics. First it presents the training of a deep neural network and then, taking advantage of the structure of the neural network,
it proposes a retrofitting policy using a modified variable importance algorithm called LIME. Considering that the first part of the study requires training data,
and that the size of the files of training data are too big to be attached to github, we will present some example codes to introduce the results of the first section
instead of the original codes.

The files of this repository are organized in the following way:

1- Trainin_Example.py : Sample script that trains a neural network given a reduced training set.
2- model_fitted_resampled_12732.pkl.zip: .pkl file that contains the final neural network model that predict the traffic performance metric.
4- LIME_Algorithm: Simplified implementation of the modified LIME algorithm that shows as a result a ranking of bridges to be retrofitted.
4- Evaluate_performance.py: Script that evaluates performance of different retrofitting strategies using the neural network.
5- util.py: auxiliar code used to compute network performance

In addition to the previous codes there are two attached folders. One is "input" which has the files needed to run the experiments. The other one is "Figures" which is a plot that has the codes necessary to run some of the experiments and respective figures of the article.

The first author Rodrigo Silva-Lopez is open to share all codes and training data upon request to the email rsilval at stanford.edu . Note that the size of all files involved in the calculation of this paper are bigger than 38 gb of data.

Some files are compressed as .zip due to limits imposed by github. In order for the codes to run properly you need to unzip those files. Some of these files are in the "input" folder
