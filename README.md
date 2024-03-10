# Filatova-Final-Project-LSML-2

In this project pre-trainied model for Yes/No Question Answering will is deployed as a Docker microservice. The work of a microservice is parallelized by distributing the load into several threads via Load balancer.

Task: Yes/No Question Answering.

Input: parameters for a model's training (batch size, lr, eps and epochs). Example: params = {"batch_size": 32, "lr": 5e-5, "eps": 1e-8, "epochs": 2}.

Output: predictions and accuracy score. 

Dataset: BoolQ. 

Model: bert-base-multilingual-cased (https://huggingface.co/bert-base-multilingual-cased). 

Experiment tracking: Neptune.ai.

Serving: Docker-compose. 

We are going to work with the BoolQ dataset from SuperGLUE. BoolQ is a question answering dataset for yes/no. In the task model training is performed on train.jsonl data and evaluation process is performed on val.jsonl. Both datasets include question, passage (question can be answered according to the passage) and label (1 (yes) or 0 (no). The data is preprocessed with BERT embeddings (tokenizer).

The data from train.jsonl was splitted into train and dev (dev_size = 10%) for evaluation during a training loop (fine-tuning BERT). Model was fine-tuned according to the following tutorial: https://mccormickml.com/2019/07/22/BERT-fine-tuning/ . Accuracy score and predictions will be computed on the validation data. 


The final ML model will be deployed in a Docker container, which will be accessed via the HTTP protocol via a POST request (REST API architectural style). The resulting microservice will be parallelized through an Nginx-based load balancer.

The file 'main.py' includes all the necessary code for loading datasets, preprocessing data, experiment tracking and model versioning in Neptune.ai, the training and testing processes. The Flask web framework is used for creating microservice REST API. The microservice takes parameters for a model's training (batch size, lr, eps and epochs) and returns the predictions and accuracy score. All the parameters and metrcis are tracked in the Neptune.ai project with each run. Examples of such runs and model's version are presented in the file 'Examples of runs and model versions in Neptune.pdf'. 

All processes are separated by containers. Nginx needs to forward requests to port 5000 in order to make the load balancer to work (code in the file 'nginx.conf'). In the file 'docker-compose.yml' the process of parallelizing the previously created microservice (name: 'test_params-microservice') using the replicas parameter is presented. The service is connected to port 5000 inside the docker virtual network, at the same time nginx-balancer redirects traffic from port 4000 to port 5000 inside the docker virtual network. For staring the processes the following command should be used: 'docker-compose up --build'. 
