# Two-Dimenstional Dichotomy

This project is an analysis of one method of optimization that was provided by Yu.Nesterov in 2017. There are theoretical results [here](https://github.com/ASEDOS999/Optimization-Halving-The-Square/blob/NetTech/One_method.pdf).

The purpose of this part is to simplify conducting an experiments with this method. For to start use this project you need to download file [user.py](https://github.com/ASEDOS999/Optimization-Halving-The-Square/blob/NetTech/NetTech/user.py) and run it. Now there are the following options:

1. LogSumExp [N=...] [time_max=...] [C=...] [eps=...]

    This function starts the comparison of our method with other inexact methods(PGM, FGM, inexact ellipsoids) on the dual problem to regularized LogSumExp-problem with linear constraints. The parameter N is a dimension of primal task. The time_max is the maximal time for experiment. The parameter C is a parameter of regularization. The parameter eps is a required accuracy for dual problem.

    The results of this command is a graph of dependence of dual value on time for different methods.

    For to use this option you have to have module matplotlib.

2. break

    This command completes work and connection with server.