# PoisoningDeepNLP

This project tries to simulate a poisoning attack on the training of a deep learning model that predicts emotions on tweets in a Federated Learning setting. 
It demonstrates how a group of malicious attackers could potentially harm the performance of a model trained collaboratively using Federated Learning by flipping the labels of the tweets. 
It also investigates the effectiveness of a simple defence called RONI (Reject on Negative Influence). 

## RONI

RONI rejects SGD updates to the model that were generated from malicious data if they impacted the accuracy negatively during that round. 
Our results show that the defense is not very effective and we need something more robust to counter poisoning attacks in a Federated Learning setting.
