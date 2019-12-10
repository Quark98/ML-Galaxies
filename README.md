# ML-Galaxies
Davit Endeladze and Leo Oliver
Predicting Galaxy Mass and Star Formation Rate Using Machine Learning

This repository contains the most important pieces of code for our semester 1 MPhys project. There are three files.
The production code imports all the raw data and performs cuts. It also visualises the features of the data used. It loads in the mass model and the SFR model and outputs the results. It also uses UMAP to reduce the dimensions of the data.
The Galaxy subclass DNN trains and tests the DNN model using balanced and unbalanced data sets and helps find the more appropriate one.
Application attempts to use the models to assign masses and SFRs to 111 million unlabelled sources.
