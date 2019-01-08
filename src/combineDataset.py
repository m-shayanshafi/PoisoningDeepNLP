import os
import math

# angerPath = 'emotionDataSet/train/anger-ratings-0to1.train.txt'
# fearPath = 'emotionDataSet/train/fear-ratings-0to1.train.txt'
# joyPath = 'emotionDataSet/train/joy-ratings-0to1.train.txt'
# sadnessPath = 'emotionDataSet/train/sadness-ratings-0to1.train.txt'

angerPath = 'emotionDataset2/train/train_anger.tsv'
fearPath = 'emotionDataset2/train/train_fear.tsv'
joyPath = 'emotionDataset2/train/train_joy.tsv'
sadnessPath = 'emotionDataset2/train/train_sadness.tsv'
surprisePath = 'emotionDataset2/train/train_surprise.tsv'
lovePath = 'emotionDataset2/train/train_love.tsv'

outfilePath = 'emotionDataset2/emotion_train.tsv' 
clientSplitPath = 'emotionDataset2/client_train/'
clientSplitPath2 = 'emotionDataset2/client_train_noniid/'

emotionSplitPath = 'emotionDataset2/train'

emotions = ['love', 'surprise', 'anger', 'sadness', 'fear', 'joy']

toLabels = {"fear":"joy" , "joy":"fear", "love":"anger", "anger":"love", "surprise":"sadness", "sadness":"surprise"}


# def completeIIDSplit():

#   angerFile = open(angerPath, 'r')
#   fearFile = open(fearPath, 'r')
#   joyFile = open(joyPath, 'r')
#   sadnessFile = open(sadnessPath, 'r')
#   surpriseFile = open(surprisePath, 'r')
#   loveFile = open(lovePath, 'r')


#   with open('./emotionDataset2/emotion_train.tsv', 'w') as outfile:

#       lineAnger = angerFile.readline()
#       lineFear = fearFile.readline()
#       lineJoy = joyFile.readline()
#       lineSadness = sadnessFile.readline()
#       lineSurprise = surpriseFile.readline()
#       lineLove = loveFile.readline()

#       while lineAnger or lineFear or lineJoy or lineSadness or lineSurprise or lineLove:
			
#           if lineAnger:
#               # lineAngerProcessed = removeLastColumn(lineAnger)
#               outfile.write(lineAnger)


#           if lineFear:
#               # lineFearProcessed = removeLastColumn(lineFear)
#               outfile.write(lineFear)


#           if lineJoy: 
#               # lineJoyProcessed = removeLastColumn(lineJoy)
#               outfile.write(lineJoy)


#           if lineSadness:
#               # lineSadnessProcessed = removeLastColumn(lineSadness)
#               outfile.write(lineSadness)

#           if lineSurprise:
#               # lineSurpriseProcessed = removeLastColumn(lineSurprise)
#               outfile.write(lineSurprise)

#           if lineLove:
#               # lineLoveProcessed = removeLastColumn(lineLove)
#               outfile.write(lineLove)


#           lineAnger = angerFile.readline()
#           lineFear = fearFile.readline()
#           lineJoy = joyFile.readline()
#           lineSadness = sadnessFile.readline()
#           lineSurprise = surpriseFile.readline()
#           lineLove = loveFile.readline()



def countRecords(file):

	thisLine = file.readline()

	numRows = 0

	while thisLine:

		numRows = numRows + 1
		thisLine = file.readline()

	return numRows


def completeIIDSplit(numClients, clientSplitPath):

	angerFile = open(angerPath, 'r')
	fearFile = open(fearPath, 'r')
	joyFile = open(joyPath, 'r')
	sadnessFile = open(sadnessPath, 'r')
	surpriseFile = open(surprisePath, 'r')
	loveFile = open(lovePath, 'r')

	angryRecords = countRecords(angerFile)
	fearRecords = countRecords(fearFile)
	joyRecords = countRecords(joyFile)
	sadnessRecords = countRecords(sadnessFile)
	surpriseRecords = countRecords(surpriseFile)
	loveRecords = countRecords(loveFile)

	angerFile = open(angerPath, 'r')
	fearFile = open(fearPath, 'r')
	joyFile = open(joyPath, 'r')
	sadnessFile = open(sadnessPath, 'r')
	surpriseFile = open(surprisePath, 'r')
	loveFile = open(lovePath, 'r')

	# lineAnger = angerFile.readline()

	# while lineAnger:


	#Splitting anger
	rowsPerClient = int(angryRecords/numClients)
	
	for client in range(0,numClients):

		clientFilePath = clientSplitPath + 'train_' + str(client) + '.tsv'

		with open(clientFilePath, 'w') as clientfile:
			
			headerRow = 'text' + '\t' + 'emotions\n'
			clientfile.write(headerRow)

			for x in range(0,rowsPerClient):
				row = angerFile.readline()
				clientfile.write(row)


	#Splitting fear
	rowsPerClient = int(fearRecords/numClients)

	for client in range(0,numClients):

		clientFilePath = clientSplitPath + 'train_' + str(client) + '.tsv'

		with open(clientFilePath, 'a') as clientfile:

			for x in range(0,rowsPerClient):
				row = fearFile.readline()
				clientfile.write(row)

	#Splitting joy
	rowsPerClient = int(joyRecords/numClients)

	for client in range(0,numClients):

		clientFilePath = clientSplitPath + 'train_' + str(client) + '.tsv'

		with open(clientFilePath, 'a') as clientfile:

			for x in range(0,rowsPerClient):
				row = joyFile.readline()
				clientfile.write(row)

	#Splitting sadness
	rowsPerClient = int(sadnessRecords/numClients)

	for client in range(0,numClients):

		clientFilePath = clientSplitPath + 'train_' + str(client) + '.tsv'

		with open(clientFilePath, 'a') as clientfile:

			for x in range(0,rowsPerClient):
				row = sadnessFile.readline()
				clientfile.write(row)

	#Splitting joy
	rowsPerClient = int(surpriseRecords/numClients)

	for client in range(0,numClients):

		clientFilePath = clientSplitPath + 'train_' + str(client) + '.tsv'

		with open(clientFilePath, 'a') as clientfile:

			for x in range(0,rowsPerClient):
				row = surpriseFile.readline()
				clientfile.write(row)

	#Splitting love
	rowsPerClient = int(loveRecords/numClients)

	for client in range(0,numClients):

		clientFilePath = clientSplitPath + 'train_' + str(client) + '.tsv'

		with open(clientFilePath, 'a') as clientfile:

			for x in range(0,rowsPerClient):
				row = loveFile.readline()
				clientfile.write(row)







	# lineAnger = angerFile.readline()

	# while lineAnger:



	# with open('./emotionDataset2/emotion_train.tsv', 'w') as outfile:

	#   lineAnger = angerFile.readline()
	#   lineFear = fearFile.readline()
	#   lineJoy = joyFile.readline()
	#   lineSadness = sadnessFile.readline()
	#   lineSurprise = surpriseFile.readline()
	#   lineLove = loveFile.readline()





	#   while lineAnger or lineFear or lineJoy or lineSadness or lineSurprise or lineLove:
			
	#       if lineAnger:
	#           # lineAngerProcessed = removeLastColumn(lineAnger)
	#           outfile.write(lineAnger)


	#       if lineFear:
	#           # lineFearProcessed = removeLastColumn(lineFear)
	#           outfile.write(lineFear)


	#       if lineJoy: 
	#           # lineJoyProcessed = removeLastColumn(lineJoy)
	#           outfile.write(lineJoy)


	#       if lineSadness:
	#           # lineSadnessProcessed = removeLastColumn(lineSadness)
	#           outfile.write(lineSadness)

	#       if lineSurprise:
	#           # lineSurpriseProcessed = removeLastColumn(lineSurprise)
	#           outfile.write(lineSurprise)

	#       if lineLove:
	#           # lineLoveProcessed = removeLastColumn(lineLove)
	#           outfile.write(lineLove)


	#       lineAnger = angerFile.readline()
	#       lineFear = fearFile.readline()
	#       lineJoy = joyFile.readline()
	#       lineSadness = sadnessFile.readline()
	#       lineSurprise = surpriseFile.readline()
	#       lineLove = loveFile.readline()






def removeLastColumn(line):

	splitLine = line.split("\t")
	# splitLine.pop()
	processedLine = splitLine[0]
	splitLine.pop(0)
	for linePart in splitLine:
		processedLine = processedLine + "\t"
		processedLine = processedLine + linePart            

	# processedLine = processedLine+"\n"

	print(processedLine)
	return processedLine

def flipLastColumn(line, fromLabel, toLabel):

	splitLine = line.split("\t")
	splitLine[-1] = splitLine[-1].replace("\r\n","")
	splitLine[-1] = splitLine[-1].replace("\n","")
	print(splitLine)

	# fromLabel = fromLabel +'\n'
	# toLabel = toLabel + '\n'

	if (splitLine[-1] == fromLabel):
		
		splitLine[-1] = toLabel

	elif splitLine[-1]==toLabel:

		splitLine[-1] = fromLabel


	processedLine = splitLine[0]
	splitLine.pop(0)
	for linePart in splitLine:
		processedLine = processedLine + "\t"
		processedLine = processedLine + linePart


	#TODO
	return processedLine

def splitDatasets(numClients, trainFilePath):

	numRows = 0

	with open(trainFilePath, 'r') as trainFile:
		
		thisLine = trainFile.readline()     

		while thisLine:

			numRows = numRows + 1
			thisLine = trainFile.readline()

	print(numRows)

	trainFile = open(trainFilePath, 'r')

	rowCount = 0
	rowsPerClient = int(numRows/numClients)

	for client in range(0,numClients):
		
		clientFileName = 'train_'+str(client)+'.tsv'
		
		with open(clientSplitPath+clientFileName, 'w') as clientfile:

			headerRow = 'text' + '\t' + 'emotions\n'
			clientfile.write(headerRow)

			for x in range(0,rowsPerClient):
				row = trainFile.readline()
				print(row)
				clientfile.write(row)

def corruptClient(clientNo, splitDir, fromLabel=None, toLabel=None):

		# dataset = argv[1]
		filename = splitDir + 'train_'+str(clientNo)+'.tsv'
		corruptedFile = splitDir + 'train_'+str(clientNo) +'_corrupted'+'.tsv'
		# header = "text" + "\t" +"emotions"
		# dir = "data/" + dataset + "/"


  #       print("Corrupting client:" + str(clientNo))        

		with open(corruptedFile, 'w') as corruptFile:

			# corruptFile.write(header + "\n")

			with open(filename, 'r') as clientfile:

				thisLine = clientfile.readline()
				corruptFile.write(thisLine)
				thisLine = clientfile.readline()
				if fromLabel==None:                     
						splitLine = thisLine.split("\t")
						fromLabel = splitLine[-1]
						fromLabel = fromLabel.replace("\r\n", "")
						toLabel = toLabels[fromLabel]
				while thisLine:
					flippedLine = flipLastColumn(thisLine, fromLabel, toLabel)

					corruptFile.write(flippedLine + "\n")

					thisLine = clientfile.readline()

		# print("Generated corrupted file:" + str(corruptFile))

def completeNonIIDSplit(numClients, outputPath, emotionDist):

	# clientPerEmotion = int(numClients/len(emotions))
	# print(clientPerEmotion)

	clientCount = 0


	for emotion in emotions:

		filePath = emotionSplitPath + "/train_" + emotion + ".tsv"
		emotionFile = open(filePath, 'r')
		emotionRecords = countRecords(emotionFile)
		emotionFile = open(filePath, 'r')
		rowsPerClient = int(emotionRecords/emotionDist[emotion])

		for client in range(0, emotionDist[emotion]):

			clientFilePath = outputPath + 'train_' + str(clientCount) + '.tsv'

			with open(clientFilePath, 'w') as clientfile:

				headerRow = 'text' + '\t' + 'emotions\n'
				clientfile.write(headerRow)

				for x in range(0,rowsPerClient):
					row = emotionFile.readline()
					clientfile.write(row)

			clientCount += 1


				
def getDataDistribution(numClients):

	emotionCounts = []
	emotionDist = {}

	for emotion in emotions:

		filePath = emotionSplitPath + "/train_"+ emotion + ".tsv"
		emotionFile = open(filePath, 'r')
		emotionRecords = countRecords(emotionFile)
		emotionCounts.append(emotionRecords)
		emotionDist[emotion] = emotionRecords

	totalRecords = sum(emotionCounts)

	emotionWeights = [emotionCount/(totalRecords*1.0) for emotionCount in emotionCounts]
	clientsPerEmotion = [int(round(numClients * emotionWeight))  for emotionWeight in emotionWeights]
	# clientsPerEmotion = [1.0 if client == 0 else client for client in clientsPerEmotion]
	count = 0
	for emotion in emotionDist:

		if clientsPerEmotion[count] == 0:
			clientsPerEmotion[count] += 1
		
		emotionDist[emotion] = clientsPerEmotion[count]
		count+=1


	return emotionDist



emotionDist = getDataDistribution(10)
print(emotionDist)

# completeIIDSplit(10, clientSplitPath)

# completeNonIIDSplit(10, clientSplitPath2, emotionDist)


# # trainPath = 'emotionDataSet'


# # splitDatasets(10, outfilePath)

# # Corrupt clients
# for x in range(0,10):
# 	corruptClient(x, clientSplitPath2, None, None)
	


	




