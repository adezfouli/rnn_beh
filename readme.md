## ActionFlow: Recurrent neural networks for the analysis of decision-making data 

This repository contains the code for the model introduced in the following paper:

Dezfouli A, Griffiths K, Ramos F, Dayan P, Balleine BW (2019). 
Models that learn how humans learn: The case of decision-making and its disorders. 
PLoS Comput Biol 15(6): e1006903

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006903

The model is called ActionFlow and describes how to fit a recurrent neural network to behavioural data,
and how to interpret it to gain insights into the underlying decision-making 
processes in the brain.

See also the following paper which uses a similar approach to fit a 
recurrent neural network to both behavioural and neural data:

Dezfouli, A., Morris, R., Ramos, F. T., Dayan, P., & Balleine, B. (2018). 
Integrated accounts of behavioral and neuroimaging data using 
flexible recurrent neural network models. 
In Advances in Neural Information Processing Systems 

https://www.biorxiv.org/content/biorxiv/early/2018/12/20/328849.full.pdf

(Note this repository does not contain the code for this paper). 

### Installation
The required packages are in `src/requirements.txt`. Using `Virtual Environments`, 
they can be installed as follows:
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
cd src
pip install -r requirements.txt
```
### Content
The repository contains three main packages:

* `actionflow`: This is the main package, which can be used for training, testing
and simulating the model. 

* `BD`: This contains the analysis of the dataset reported in the paper. It uses 
`actionflow` for training, testing and simulating the model. There are two
packages in `BD`: one is `fit` and the other one is `sim`. `fit` contains 
the files used for fitting the models to data, and `sim` contains the files 
used for simulating and testing the model. A brief comment is
presented at the beginning of each file in this package to summarise its content. 

    For example,
    for running cross-validation experiments, the following command can be used
    (current directory should be `src`):
 
    ```python -m BD.fit.rnn_cv n_proc```
    
    `n_proc` number of parallel processes
    
    The package also contains an `R` folder which contains the codes 
    used for data analysis and generating graphs in the paper.

* `examples`: This includes an IPython notebook to demonstrate the basic functionalities
of `actionflow`.
 