from fedLearn import preprocessData, constructGlobalVocab
from sklearn.model_selection import train_test_split
import numpy as np
import dataLoadUtils
from torch.utils.data import DataLoader
import sys
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import emotionModel
from copy import deepcopy

iid = False
dataDirIID = 'emotionDataset2/client_train'
dataDirNonIID = 'emotionDataset2/client_train_noniid'
globalTrainPath = 'emotionDataset2/client_train/global_train.tsv'
num_emotions = 6
emotion_dict = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
numClients = 10
maliciousClients = 0
attack_name = 'fear_joy'
roni = True
RONI_THRESH = 2
numVerifiers = 3
calibrationSetSize = 0.2
TRAIN_QUIZ_SPLIT = 0.2
BATCH_SIZE = 64
TRAIN_QUIZ_PAIRS = 2
PATH = 'modelFile'
calibratedThresh = True


def getTrainQuizSets(beforeInputSet, beforeTargetSet, TRAIN_QUIZ_PAIRS):

	totalSamples = len(beforeInputSet)
	print(totalSamples)
	sampleSize = totalSamples/TRAIN_QUIZ_PAIRS

	# print(sampleSize)

	trainSets = []
	quizSets = []

	for x in xrange(0,TRAIN_QUIZ_PAIRS):

		sampleIdxs = np.random.choice(totalSamples, sampleSize, replace=False) # Replace False because you don't want repeats in the sample
		sampleInputTensors = [beforeInputSet[idx] for idx in sampleIdxs]
		# print(len(sampleInputTensors))    
		sampleTargetTensors = [beforeTargetSet[idx] for idx in sampleIdxs]
		# print(BATCH_SIZE)
	
		trainInput, quizInput, trainTarget, quizTarget = train_test_split(sampleInputTensors, sampleTargetTensors, test_size=TRAIN_QUIZ_SPLIT)
	
		trainDataset = dataLoadUtils.MyData(trainInput, trainTarget)
		trainDataset = DataLoader(trainDataset, batch_size = BATCH_SIZE, drop_last=True,shuffle=True)
	
		quizDataset =  dataLoadUtils.MyData(quizInput, quizTarget)
		quizDataset = DataLoader(quizDataset, batch_size = BATCH_SIZE, drop_last=True,shuffle=True)

		# print(len(trainDataset))

		trainSets.append(trainDataset)
		quizSets.append(quizDataset)

	return trainSets, quizSets



def main():

	globalVocab = constructGlobalVocab()
	validationFileName = 'val.tsv'
	input_tensor_val, target_tensor_val = preprocessData(validationFileName, globalVocab, iid=iid)

	# Creating training and validation sets using an 80-20 split
	beforeInputSet, calibrateInputSet, beforeTargetSet, calibrateTargetSet = train_test_split(input_tensor_val, target_tensor_val, test_size=calibrationSetSize)

	trainSets, quizSets = getTrainQuizSets(beforeInputSet, beforeTargetSet, TRAIN_QUIZ_PAIRS)

	calibrateDataset = dataLoadUtils.MyData(calibrateInputSet, calibrateTargetSet)
	calibrateDataset = DataLoader(calibrateDataset, batch_size = BATCH_SIZE, drop_last=True,shuffle=True)


	trainingIterations = 200

	vocab_inp_size = len(globalVocab.word2idx)
	embedding_dim = 256
	units = 1024
	target_size = num_emotions

	models = []
	optimizers = []

	criterion = nn.CrossEntropyLoss() # the same as log_softmax + NLLLoss

	for idx in xrange(0,TRAIN_QUIZ_PAIRS):	
	
		use_cuda = True if torch.cuda.is_available() else False
		device = torch.device("cuda" if use_cuda else "cpu")
		model = emotionModel.EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
		model.to(device)
		models.append(model)
		optimizer = torch.optim.Adam(models[idx].parameters())
		optimizers.append(optimizer)

	VAL_BUFFER_SIZE = len(input_tensor_val)
	VAL_N_BATCH = VAL_BUFFER_SIZE // BATCH_SIZE

	print(len(calibrateDataset))

	diffAccuracies = np.zeros([trainingIterations, TRAIN_QUIZ_PAIRS, len(calibrateDataset)])
	print(diffAccuracies.shape)
	trainIterator = [(iter(dataset)) for dataset in trainSets]


	for iteration in xrange(0,trainingIterations):

		# for quizSet in quizSets:

		# Measure current validation accuracy

		# prev_val_accuracy = 0
		# numBatches = 0

		# for (batch, (inp, targ, lens)) in enumerate(quizSets[0]):

		#   predictions,_ = model(inp.permute(1, 0).to(device), lens, device)        
		#   batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
		#   prev_val_accuracy += batch_accuracy
		#   numBatches += 1


		
		for setIdx in xrange(0, TRAIN_QUIZ_PAIRS):
			
			prev_val_accuracy = measureAccuracy(models[setIdx],quizSets[setIdx], device)
			print("Model %d accuracy at iteration %d: %d" % (setIdx, iteration,prev_val_accuracy))

			batchCount = 0

			currentState = deepcopy(models[setIdx].state_dict())


			for (batch, (inp, targ, lens)) in enumerate(calibrateDataset):

				loss = 0
				predictions, _ = models[setIdx](inp.permute(1 ,0).to(device), lens, device) # TODO:don't need
				loss += emotionModel.loss_function(targ.to(device), predictions)
				batch_loss = (loss / int(targ.shape[1]))

				optimizers[setIdx].zero_grad()
				loss.backward()

				optimizers[setIdx].step()
				# print("Printing after grad")

				# for name,param in model.named_parameters():
				# 	print(param.data)
				# 	break

				newAccuracy = measureAccuracy(model, quizSets[setIdx], device)

				# print("Model %d accuracy at iteration %d: %d" % (setIdx, iteration,prev_val_accuracy))
				print("Accuracy after calibration batch  %d: %d" % (batchCount,newAccuracy))

				diff = newAccuracy - prev_val_accuracy
				diffAccuracies[iteration,setIdx, batchCount] = diff 

				# print("Printing after undoing grad")
				models[setIdx].load_state_dict(currentState)         

				# for name,param in model.named_parameters():
				#   print(param.data)
				#   break


				batchCount = batchCount + 1

			# print(diffAccuracies)


			nextBatch = next(trainIterator[setIdx])

			inp = nextBatch[0]
			targ = nextBatch[1]
			lens = nextBatch[2]

			loss = 0
			predictions, _ = models[setIdx](inp.permute(1 ,0).to(device), lens, device) # TODO:don't need
			loss += emotionModel.loss_function(targ.to(device), predictions)
			batch_loss = (loss / int(targ.shape[1]))       
			
			optimizers[setIdx].zero_grad()

			loss.backward()

			optimizers[setIdx].step()











	# 		# print(model.parameters())
	# 		# sys.exit(0)





	# 		# model = torch.load(PATH)

	# 		# getData
	# 		# getGrad(model)
	# 		# optimizer.step()

	print(diffAccuracies)
	
	iterThreshs = []

	for iteration in range(0,trainingIterations):

		iterationDiffs = diffAccuracies[iteration, :, :] 
		print(iterationDiffs.shape)
		iterationDiffsAvg = np.mean(iterationDiffs, axis=0)
		iterThresh = iterationDiffsAvg.mean() - 3 * iterationDiffsAvg.std()
		iterThreshs.append(iterThresh)

	print(iterThreshs)



	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# model = EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
	# model.to(device)

def measureAccuracy(model,dataset, device):

	accuracy = 0
	numBatches = 0

	for (batch, (inp, targ, lens)) in enumerate(dataset):

		predictions,_ = model(inp.permute(1, 0).to(device), lens, device)        
		batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
		accuracy += batch_accuracy
		numBatches += 1

	accuracy = (accuracy / numBatches*1.0)

	return accuracy

if __name__ == "__main__":
	main()




















# print("here")

# print("here")


