# ECGN
This is our Tensorflow implementation for our ECGN 2021 paper(under review) and a part of baselines:

>Bin Wu, Lihong Zhong, Hui Li and Yangdong Ye(2021). Efﬁcient Complementary Graph Convolutional Network without Negative Sampling for Item Recommendation(under review).

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.16.4
* scipy == 1.3.1
* pandas == 0.17

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.
NOTE: The cpp implementation is much faster than python.**

## Examples to run ECGN:
run [main.py](./main.py) in IDE or with command line:
```
python main.py
```

NOTE :   
(1) the duration of training and testing depends on the running environment.  
(2) set model hyperparameters on .\conf\ECGN.properties  
(3) set NeuRec parameters on .\NeuRec.properties  
(4) the log file save at .\log\office\  

## Dataset
We provide office dataset.
  * .\dataset\office.train, office.test and office.ii
  *  Each line is a user with her/his positive interactions with items: userID \ itemID \ ratings .
  *  Each user has more than 5 associated actions.

## Baselines
The list of available models in EAGCN, along with their paper citations, are shown below:

| General Recommender | Paper                                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| BPRMF               | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.                   |
| ENMF                | Chong Chen al., Efficient neural matrix factorization without sampling forrecommendation. Trans 2020.         |
| WRMF                | Yifan Hu,  al., Collaborative filtering for implicit feedback datasets. ICDM 2020.                            |
| NGCF                | Xiang Wang, et al., Neural Graph Collaborative Filtering. SIGIR 2019.                                         |
| LightGCN            | Xiangnan He, et al., LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.|
