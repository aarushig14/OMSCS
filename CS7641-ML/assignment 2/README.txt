Datasets -
------------------------

1. Wine Quality Dataset - https://archive.ics.uci.edu/ml/datasets/wine+quality

Implemented using python 3.9 - sklearn, matplotlib, numpy, pandas. Use pip to install any missing libraries.
Download mlrose-hiive from https://pypi.org/project/mlrose-hiive/
>> pip install mlrose-hiive

Two python code files -
-------------------------

1. main.py 
	- ensure that following flags are set True to generate the graphs in report 
		plot_individuals = True // individual comparison shared in Section 3
		plot_comparison = True // comparison among algorithms for optimisation problems shared in Section 2
2. neural_network_optimised.py - used to generate table for weighted optimisations performed in neural network machine learning algorithm.

How to run python files -
--------------------------

python main.py
python neural_network_optimised.py

How results are stored -
--------------------------

Outputs are stored for each algorithm in their own folder parallel to python files. 
Comparison graphs and function evaluation graph are stored FITNESS_CURVES.
output.csv stores the summary from main.py
CSV files for each nn version is stored in respective algorithm folder

