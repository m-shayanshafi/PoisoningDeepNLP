import pdb
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re

colors = ['black', 'red', 'green', 'blue', 'yellow']
labels = ['30% poisones', 'RONI - Calibrated Centralized']

def plotResults(outputFile, inputFiles):

	print outputFile

	fig, ax = plt.subplots(figsize=(10, 5))
	toplot = np.zeros((len(inputFiles), 200))

	# unpoisonedFedlearn = "results_FedLearn.csv"

	lines = []

	fileIdx = 0
	for inputFile in inputFiles:

		print inputFile
		df =  pd.read_csv(inputFile, header=None)	
		toplot[fileIdx] = 100 - df[1].values
		fileIdx+=1

	lineIdx = 0

	for dataPoints in toplot:
		
		thisLine =  mlines.Line2D(np.arange(200), dataPoints, color=colors[lineIdx],	linewidth=3, linestyle='-', label=labels[lineIdx])	
		ax.add_line(thisLine)
		lines.append(thisLine)
		lineIdx+=1

	# for line in lines:



	# ###########################################
	
	# unpoisonedDF = pd.read_csv(unpoisonedFedlearn, header=None)
	# toplot[0] = 100 - unpoisonedDF[1].values
	# l1 = mlines.Line2D(np.arange(200), toplot[0], color='black', 
	# 	linewidth=3, linestyle='-', label="Federated Learning")	

	# ###########################################

	# ax.add_line(l1)
	plt.legend(handles=lines, loc='best', fontsize=18)
	
	axes = plt.gca()	

	plt.ylabel("Validation Error", fontsize=22)
	axes.set_ylim([0, 100])

	plt.xlabel("Training Iterations", fontsize=22)
	axes.set_xlim([0, 200])

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)

	fig.tight_layout(pad=0.1)

	fig.savefig(outputFile)

def parseResults(fname):

	# fname = input_file_directory + "/LogFiles_" + str(i) + "/log_0_" + str(total_nodes)  + ".log"
	lines = [line.rstrip('\n') for line in open(fname+".csv")]
	# if not os.path.exists(outname):
	# 		os.makedirs(output_file_directory)
	outfile = open(fname + "_parsed.csv" , "w")

	for line in lines:

		idx = re.match('^[\\d]*,[\\d]*.[\\d]*,[\\d]*.[\\d]*',line)
		
		if idx:
			outfile.write(line+"\n")

def getRejectedUpdates(fname, numClients):

	lines = [line.rstrip('\n') for line in open(fname+".csv")]

	numberAccepted = np.zeros(numClients)
	numberRejected = np.zeros(numClients)

	count = 0

	for line in lines:

		if line == "True" or line == "False":
			
			idx = count % numClients
			if line == "True":
				numberAccepted[idx] += 1
			if line == "False":
				numberRejected[idx] += 1

			count = count + 1

	return numberAccepted, numberRejected


if __name__ == '__main__':

	plotResults('eval_iid_calibrated_30.jpg', ['results_poison_fearjoy_3.csv', 'iid_roni_calibrated_30_parsed.csv'])

	# parseResults("iid_roni_calibrated_30")

	# Get rejected updates by each client
	# accepted, rejected = getRejectedUpdates('roni_centralized_iid', 10)
	# print(accepted)
	# print(rejected)
