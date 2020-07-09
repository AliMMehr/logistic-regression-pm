# Logistic Regression Plus-Minus

This repository contains the codes for _David Poole, Ali Mohammad Mehr, Wan Shing Martin Wang_ ,"Conditioning on “and nothing else”: Simple Models of Missing Data betweenNaive Bayes and Logistic Regression," ICML 2020 Workshop Artemiss Submission.

## running the code
1. `pip install -r requirements.txt`

2. `./run_tests.sh 1 200 10`

## explanations for above commands
To run the code, first install the dependencies using `pip install -r requirements.txt` .

Next, run `./run_tests.sh 1 200 10` to run the tests and get the output graphs. As you can see, this command accepts 3 arguments as follows:

- start_id: The id of the first test. Suggested value: 1

- end_id: The id of the last test. Suggested value: 200

- number of concurrent tests running. Suggested value: 10 (if your system has at least 10 CPU threads)

The script will run $$(end\_id-start\_id+1)$$ tests. Each test is composed of creating 3 randomly generated datasets - namely a C dataset, a D4 dataset, and a DLR dataset. On each generated dataset, the code traines the four different models mentioned in our paper - namely models LR$$\pm$$, Naive Bayes, model (c), and model (d). For each generated dataset, the logloss of each model for that dataset is stored in an individual json file in _results/_ directory.

Finally, the script will run `python Read-test-results-and-plot-graphs.py` to make the graphs shown in paper alongs with printing average logloss comparisons.
