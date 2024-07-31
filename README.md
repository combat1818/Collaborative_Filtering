
# Collaborative Filtering Project

Semester project for the Computational Intelligence Lab 2024 at ETH Zurich. This repository contains the implementation of the collaborative filtering methods developed by the "CilTeam" team. The results were submitted to the in-class kaggle competition.

## Overview
We implement multiple base models including:
- SVD
- Alternating Least Squares
- IterativeSVD
- RSVD
- SVD++
- KNN Means
- Autoencoder
- Neural Collaborative Filtering
- Deep Matrix Factorization

We also perform blending and stacking of different combinations of base models using GradientBoosting, XGB and AdaBoost algorithms. The details about the results can be found in the report. 

## Dependencies
There are several python dependencies required to run the code. They can be installed by running:
```
npm install numpy pandas torch scikit-learn xgboost
```

## Reproducibility
After installing the dependencies you need to navigate to the top folder:
```
git clone 
cd Collaborative_Filtering
```

You can train a single model and evaluate it on a hold-out test set by running:
```
python "main.py file path" --model "SVDpp" --train
```
or train a model on the full training set and generate predictions for submission:
```
python "main.py file path" --model "SVDpp" --submission
```

Furthermore you can blend/ stack multiple base models and train a meta_model using:
```
python -u "main.py file path" --stacking --meta_model "XGB" --SVD 
--ALS --IterativeSVD --RSVD --SVDpp --KNNMeans --train
```

Analogically by changing the `--train` flag to `--submission` flag you generate the submission file. Change `--stacking` to `--blending` to blend the base models instead. You can choose any base models by adding the flags from the list:
- `--SVD`
- `--ALS`
- `--IterativeSVD`
- `--RSVD`
- `--SVDpp`
- `--KNNMeans`
- `--Autoencoder`
- `--NCF`
- `--DMF`
You can choose the meta model from: `"XGB"`, `"GB"` and `"AdaBoost"`. Training the models saves the intermediate out-of-fold predictions to a file to speed-up future computations. By default the predictions for base models are reused so if you want any of the base models to be retrained (because for example you changed its hyperparameters) you need to remove its corresponding predictions from the `\predictions` folder. 

## Authors
Aleksander Lorenc, Artur Zolkowski and Madeleine Sandri