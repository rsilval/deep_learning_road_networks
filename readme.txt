This folder contains the code associated to the study: "Deep learning retrofittting and risk assessment of road networks"
by Rodrigo Silva-Lopez, Jack W. Baker and Alan Poulos

The paper presents two main topics. First it presents the training of a deep neural network and then, taking advantage of the structure of the neural network,
it proposes a retrofitting policy using a modified variable importance algorithm called LIME. Considering that the first part of the study requires training data,
and that the size of the files of training data are too big to be attached to github, we will present some example codes to introduce the results of the first section
instead of the original codes.

The files of this repository are organized in the following way:

1- : Sample script that trains a neural network given a reduced training set.
2- model_fitted_resampled_12732.pkl.zip: .pkl file that contains the final neural network model that predict the traffic performance metric.
3- LIME_Algorithm.py: Python file that shows a function that given a list of damaged bridges it computes the traffic performance metric.
4- : Simplified implementation of the modified LIME algorithm that shows as a result a ranking of bridges to be retrofitted.
5- : Script that evaluates performance of different retrofitting strategies using the neural network.
6- : Code that generates analogue figures to the ones shown in Section: "Neural networks to predict seismic risk"

The first author Rodrigo Silva-Lopez is open to share all codes and training data upon request to the email rsilval at stanford.edu . Note that the size of all files involved in the calculation of this paper are bigger than 38 gb of data.
