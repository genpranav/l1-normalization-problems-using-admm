
# L1-Norm Problems using ADMM
Least Absolute Deviation (LAD), Huber Fitting and Least Absolute Shrinkage and Selection Operator (LASSO) solved using the Alternating Direction Method of Multipliers (ADMM) approach, from a Linear Algebra standpoint along with the Machine Learning justification for exploration.

A loss function is the measure used to guide the learning process of Neural Networks. A regularized loss has two parts, a term that measures error on the data and a penalty on the model weights.

> Loss = data-fidelity term (prediction error) + regularization factor * norm(weights)

The type of norm used as the penalty decides the learning behaviour relative to the data encountered:

L1 = summation(|x|)

L2 = root(summation(x^2))   
(the L2 *norm*; L2 regularization commonly uses its square, summation(x^2))

L1 regularization tends to drive weights exactly to zero, which is why it doubles as a feature-selection tool, it helps answer "what are the most significant features to include?"

## Documentation

The Documentation of this project - [here](./Report.pdf)

  
## Run Locally

Feature selector results compared with individual feature correlation, in Python - [here](./LASSO_FeatureSelector.ipynb)

Loss plots:

LAD and LS, in Python - [here](./LAD_vs_LS_loss.ipynb)

L1, L2, Huber, in MATLAB - [here](./Huber_loss.mlx)

Update equation iterations, in MATLAB - [here](./LAD_Huber_LASSO_Code.mlx)
