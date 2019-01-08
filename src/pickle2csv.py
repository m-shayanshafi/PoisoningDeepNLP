import csv
from six.moves import cPickle as pickle
import numpy as np
import pandas as pd
import util
import base64
from sklearn.model_selection import train_test_split

def main(path_pickle,path_csv):

	data = pd.read_pickle(path_pickle)
	# print(data.head(10))

	with open(path_csv+"/global_train.tsv",'w') as f:

		wr = csv.writer(f, delimiter='\t')
		header = ['text', 'emotions']
		wr.writerow(header)

		for index, row in data.iterrows():
			
			line = [str(row['text']), str(row['emotions']) ]
			# print(line)
			wr.writerow(line)

	# Divide into test/train/vaild here
	data_train, data_val = train_test_split(data,  test_size=0.2)
	data_val, data_test = train_test_split(data_val, test_size=0.5)
	
	#Sanity check
	
	print(data.shape)	
	print(data_train.shape)
	print(data_val.shape)
	print(data_test.shape)

	# Divide train by emotion
	
	data_joy = data_train.loc[data_train['emotions'] == "joy"]
	data_sadness = data_train.loc[data_train['emotions'] == "sadness"]
	data_fear = data_train.loc[data_train['emotions'] == "fear"]
	data_love = data_train.loc[data_train['emotions'] == "love"]
	data_anger = data_train.loc[data_train['emotions'] == "anger"]
	data_surprise = data_train.loc[data_train['emotions'] == "surprise"]


	# Make separate file for each emotion

	# with open(path_csv+"/train_joy.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row_joy in data_joy.iterrows():
			
	# 		line_joy = [str(row_joy['text']), str(row_joy['emotions']) ]
	# 		# print(line)
	# 		wr.writerow(line_joy)

	# with open(path_csv+"/train_sadness.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row in data_sadness.iterrows():
			
	# 		line = [str(row['text']), str(row['emotions']) ]
	# 		# print(line)
	# 		wr.writerow(line)

	# with open(path_csv+"/train_fear.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row in data_fear.iterrows():
			
	# 		line = [str(row['text']), str(row['emotions']) ]
	# 		# print(line)
	# 		wr.writerow(line)

	# with open(path_csv+"/train_love.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row in data_love.iterrows():
			
	# 		line = [str(row['text']), str(row['emotions']) ]
	# 		# print(line)
	# 		wr.writerow(line)

	# with open(path_csv+"/train_anger.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row in data_anger.iterrows():
			
	# 		line = [str(row['text']), str(row['emotions']) ]
	# 		# print(line)
	# 		wr.writerow(line)

	# with open(path_csv+"/train_surprise.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row in data_surprise.iterrows():
			
	# 		line = [str(row['text']), str(row['emotions']) ]
	# 		# print(line)
	# 		wr.writerow(line)

	with open(path_csv+"/test.tsv",'w') as f:

		wr = csv.writer(f, delimiter='\t')
		header = ['text', 'emotions']
		wr.writerow(header)


		for index, row in data_test.iterrows():
			
			line = [str(row['text']), str(row['emotions']) ]
			# print(line)
			wr.writerow(line)

	with open(path_csv+"/val.tsv",'w') as f:

		wr = csv.writer(f, delimiter='\t')
		header = ['text', 'emotions']
		wr.writerow(header)
		for index, row in data_val.iterrows():
			
			line = [str(row['text']), str(row['emotions']) ]
			# print(line)
			wr.writerow(line)






	# with open(path_csv+"/train_sadness.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row_sadness in data_sadness.iterrows():
			
	# 		line_sadness = [str(row_sadness['text']), str(row_sadness['emotions']) ]
	# 		wr.writerow(line_sadness)

	# with open(path_csv+"/train_fear.tsv",'w') as f:

	# 	wr = csv.writer(f, delimiter='\t')

	# 	for index, row_fear in data_fear.iterrows():
			
	# 		line_sadness = [str(row_fear['text']), str(row_fear['emotions']) ]
	# 		wr.writerow(line_fear)


	
	


main('merged_training.pkl', 'emotionDataset2/client_train')