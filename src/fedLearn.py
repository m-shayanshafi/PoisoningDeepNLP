import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import re
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import util
import ConstructVocab as construct
import dataLoadUtils
import emotionModel
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import sys
import random

iid = True
dataDirIID = 'emotionDataset2/client_train'
dataDirNonIID = 'emotionDataset2/client_train_noniid'
globalTrainPath = 'emotionDataset2/client_train/global_train.tsv'
num_emotions = 6
emotion_dict = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
numClients = 10
maliciousClients = 3
attack_name = 'fear_joy'
roni = True
RONI_THRESH = 2
numVerifiers = 3
RONIThresh = [9.10966003863933, -15.874558874368912, -7.775880098915261, -4.844163469628378, -5.355666615271283, -20.39261443660491, -0.8901899490807281, -5.571214285111273, 0.08154215976568802, -4.604128095783668, -1.036347557766009, -3.5495038856803265, -5.577401952256601, -1.4607360936279095, -4.817526507976634, -2.1730893782122194, -2.5946828902290444, -3.9201052230777798, -2.5605140800376875, -3.8512528738145413, -2.3399486042067514, -4.08496231131986, -2.5337795048242233, -3.406825274832579, -2.882480722289613, -3.2647324177963783, -2.48892426136705, -5.271242781645216, -1.8306680136908797, -4.3184714418872, 0.17035372573013108, -2.324823136242801, -3.6769230769230767, -3.9140008970911766, -3.251647243046709, -4.2875462984961725, -5.100839091498487, -2.1063822041826015, -2.8595211753348604, 1.8247746740546127, -6.253403206039712, 0.41702615999887627, -5.505571665427756, -0.34615384615384626, -6.25637034795745, -1.8643562572741568, -3.0023731647220377, -6.176923076923078, 2.260029350914332, -5.489011805673428, -2.1215004487936255, -1.9722434782597205, -3.3062755074770522, -3.651271451418695, -1.5270976523952036, -4.7041852229635115, -2.4311707235395192, -4.463157496180687, -2.6834103126129625, -3.228939141997034, -3.016256610971489, -3.340555324753943, -2.4960792429922725, 1.1758990668975917, -2.994499819529915, -1.077803181331721, -2.6258053694541146, -0.498745448733402, -4.164247047534927, -3.859938704824879, -1.5613146734773349, -3.7870774683083717, -1.24469262819393, -4.065567873499983, -0.962177779780641, -5.031930624717546, 0.8033888920199121, -4.2565543034791045, -2.6013125572223603, -3.3927255720107583, -4.549829756285738, -1.13023124137425, -4.29742245112535, -0.3500828105030904, -4.981852018779515, -3.3373401717935414, -3.66413197812034, -4.89101983252467, -4.347858298093035, -5.489178254225197, -2.3809451137773814, -4.049734055426126, -3.2368468879001475, -1.9274217006893142, -4.097434723933334, -5.376473166008591, -3.159530317526678, -2.2934908650200465, -2.6633044692989865, -0.13698804292013111, -4.341028119615652, -2.204868174158147, -5.212375832505266, -2.855192750223102, -4.785182053280973, -4.122407558705214, -0.6769230769230767, -5.795609153540777, -4.2884751966438746, -3.208840180903265, -4.16897992323104, -3.3198178058790146, -2.276596576176571, -3.444778702037527, -3.2258835957262013, -3.9453468620613754, 0.32095947659757584, -3.527819827727452, -5.638580533452608, -5.229302081946069, -1.9822801629405067, -7.478188711427606, -4.386761147136364, -5.499890732061001, -5.799419675242175, 0.8836428611800504, -4.573645704717715, -0.846228238453679, -3.5494032277005787, -3.1912583074493, -5.412161554688228, -2.112812623255488, -3.2783567066017665, 0.16218163481393155, -2.3171045403037134, -9.381038524974137, -3.1623994668153776, -4.25410879613924, -8.716943231376995, -2.3870560213219743, -3.525714252607376, -1.0871133352432278, -2.2069071739059547, -4.147478473152252, -2.243810253611972, 0.018026951705034655, -0.16664671244026374, -3.0199497159151134, -5.654587978913355, -4.371670137085074, -9.685414094860423, -5.666032232249943, -5.562383757139002, 0.7179433514767251, 0.4084876998281417, -1.667307049138081, -3.6907758315762234, -4.936215984856279, -2.533471086330195, -4.848562575964317, -4.382024108311526, -4.990818265054027, -2.26199638873764, -7.002725920839845, -4.453622521350755, -7.972224818351831, -6.734856243406668, -4.296002125815612, -5.477512908452916, -4.418539555630095, 2.954883843212528, 2.843314464211848, 0.012368888064473005, -4.056825651843433, -10.20846743421535, -6.638057118557475, -3.4571787561132328, -1.8361042933905525, 3.2267134396390302, -1.0146743880720726, -3.1215848189612974, -7.417964956691099, -5.763730019971957, -3.875120480749781, 1.3860825334385458, -1.4708434530457866, -2.6600679970228063, -8.93065326486746, -5.2577141833737375, -3.3444646876793955, -5.062785049553408, -1.9922969036361304, -3.6695955989723066, -1.9341002062666734, -1.5726675749750354, -1.3368411071529493, -3.373198568691579, -0.20945975193114785, -8.63597343239873, -7.613200027064615]
iterationThresh = 0

embedding_dim = 256
units = 1024
vocab_inp_size = 0
target_size = num_emotions
BATCH_SIZE = 60
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def roni(currentModel, sgdUpdate, previousValError, iterationNum, val_dataset):

    updatedModel = currentModel + sgdUpdate
    tempModel = emotionModel.EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
    tempModel.to(device)

    layer = 0
    
    for name, param in tempModel.named_parameters():
        if param.requires_grad:
            param.data = torch.tensor(updatedModel[layer])
            layer += 1

    val_accuracy = 0

    batchNum = 0

    for (batch, (inp, targ, lens)) in enumerate(val_dataset):

        predictions,_ = tempModel(inp.permute(1, 0).to(device), lens, device)        
        batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
        val_accuracy += batch_accuracy
        batchNum += 1

    val_accuracy = (val_accuracy / batchNum)

    # print('{:.4f},{:.4f}'.format(val_accuracy, previousValError))

    if iterationNum < 0:

        if (previousValError - val_accuracy) <= RONI_THRESH:
            return True
        else:
            return False

    else:

        if (val_accuracy - previousValError) >= RONIThresh[iterationNum]:
            return True
        else:
            return False



def roniPeerToPeer(currentModel, sgdUpdate, peerIndices):

    tempModel = emotionModel.EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
    tempModel.to(device)

    # print(peerIndices)

    layer = 0
    for name, param in tempModel.named_parameters():
        if param.requires_grad:
            param.data = currentModel[layer]
            layer += 1

    prev_accuracies = []

    for peer in peerIndices:
        
        peerAccuracy = 0
        batchCount = 0
        for (batch, (inp, targ, lens)) in enumerate(train_dataset_clients[peer]):

            predictions,_ = tempModel(inp.permute(1, 0).to(device), lens, device)        
            batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
            # print(batch_accuracy)
            peerAccuracy += batch_accuracy
            batchCount+=1
            break

        peerAccuracy = (peerAccuracy / batchCount) 
        # print(peerAccuracy)
        # print(batchCount)
        prev_accuracies.append(peerAccuracy)

    print(prev_accuracies)
    # sys.exit()


    
    updatedModel = currentModel + sgdUpdate
    
    layer = 0
    for name, param in tempModel.named_parameters():
        if param.requires_grad:
            param.data = updatedModel[layer]
            layer += 1

    new_accuracies = []

    for peer in peerIndices:
        
        peerAccuracy = 0
        batchCount = 0
        for (batch, (inp, targ, lens)) in enumerate(train_dataset_clients[peer]):

            predictions,_ = tempModel(inp.permute(1, 0).to(device), lens, device)        
            batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
            peerAccuracy += batch_accuracy
            batchCount+=1

        peerAccuracy = (peerAccuracy / batchCount) 
        new_accuracies.append(peerAccuracy)

    print(new_accuracies)
    # sys.exit

    diffs = [prev - new for prev, new in zip(prev_accuracies, new_accuracies)]

    passes = sum(diff < RONI_THRESH for diff in diffs)

    if passes >= 2:
        return True
    else:
        return False





def getCurrentModel(model):

    modelLayers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            modelLayers.append(param.grad)

    # print(modelLayers)

    return modelLayers

def constructGlobalVocab():

    data = pd.read_csv(globalTrainPath, sep='\t')

    # # retain only text that contain less that 70 tokens to avoid too much padding
    data["token_size"] = data["text"].apply(lambda x: len(x.split(' ')))
    data = data.loc[data['token_size'] < 70].copy() 

    # # construct vocab and indexing
    inputs = construct.ConstructVocab(data["text"].values.tolist()) 

    return inputs


def pandasToTensor(data, globalVocab):

    data = shuffle(data)

    # # Preprocessing data
    # # retain only text that contain less that 70 tokens to avoid too much padding
    data["token_size"] = data["text"].apply(lambda x: len(x.split(' ')))
    data = data.loc[data['token_size'] < 70].copy() 

    # # sampling
    # data = data.sample(n=50000);

    # # construct vocab and indexing
    # inputs = construct.ConstructVocab(data["text"].values.tolist())   

    # print(globalVocab.vocab[0:10])

    input_tensor = [[globalVocab.word2idx[s] for s in es.split(' ')]  for es in data["text"].values.tolist()]

    # examples of what is in the input tensors
    # print(input_tensor[0:2])

    # calculate the max_length of input tensor
    max_length_inp = util.max_length(input_tensor)
    # print(max_length_inp)

    # inplace padding
    input_tensor = [util.pad_sequences(x, max_length_inp) for x in input_tensor]
    # print(input_tensor[0:2])

        ###Binarization
    emotions = list(emotion_dict.values())
    num_emotions = len(emotion_dict)
    # print(emotions)
    # binarizer
    mlb = preprocessing.MultiLabelBinarizer(classes=emotions)
    data_labels =  [emos for emos in data[['emotions']].values]
    # print(data_labels)
    bin_emotions = mlb.fit_transform(data_labels)
    target_tensor = np.array(bin_emotions.tolist())

    # print(target_tensor[0:2])
    # print(data[0:2]) 

    get_emotion = lambda t: np.argmax(t)

    get_emotion(target_tensor[0])   
    emotion_dict[get_emotion(target_tensor[0])]

    return input_tensor, target_tensor  


def preprocessData(filename, globalVocab, getAttackVal = False, iid = True):

    filePath = ''
    if iid:
        filePath = dataDirIID+'/'+filename
    else:
        filePath = dataDirNonIID+'/'+filename
    # print(filePath)

    # print(filePath)
    data = pd.read_csv(filePath, sep='\t')

    if iid:
        if "corrupted" in filename or ("val" in filename and getAttackVal):
            attack_label1, attack_label2 = attack_name.split('_')
            data = data.loc[(data['emotions'] == attack_label1) | (data['emotions'] == attack_label2)] 
    else:

        if ("val" in filename and getAttackVal):
            attack_label1, attack_label2 = attack_name.split('_')
            data = data.loc[(data['emotions'] == attack_label1) | (data['emotions'] == attack_label2)] 


    input_tensor, target_tensor = pandasToTensor(data, globalVocab)

    return input_tensor, target_tensor  



def main():

    globalVocab = constructGlobalVocab()

    input_tensor_clients = []
    target_tensor_clients = []

    for x in range(0,numClients):
        trainFileName = ''
        if x < (numClients - maliciousClients):
            trainFileName = 'train_' + str(x)+'.tsv'    
        else:
            trainFileName = 'train_' + str(x)+'_corrupted.tsv'
        
        input_tensor_train, target_tensor_train = preprocessData(trainFileName, globalVocab, iid=iid)
        input_tensor_clients.append(input_tensor_train)
        target_tensor_clients.append(target_tensor_train)

    validationFileName = 'val.tsv'
    input_tensor_val, target_tensor_val = preprocessData(validationFileName, globalVocab, iid=iid)

    # validationFileName = 'val.tsv'
    input_tensor_attackRate, target_tensor_attackRate = preprocessData(validationFileName, globalVocab,True, iid=iid)

    testFileName = 'test.tsv'
    input_tensor_test, target_tensor_test = preprocessData(testFileName, globalVocab, iid=iid)


    TRAIN_BUFFER_SIZE = len(input_tensor_train)
    VAL_BUFFER_SIZE = len(input_tensor_val)
    ATTACKRATE_BUFFER_SIZE = len(input_tensor_attackRate)
    TEST_BUFFER_SIZE = len(input_tensor_test)
    BATCH_SIZE = 60
    TRAIN_N_BATCH = TRAIN_BUFFER_SIZE // BATCH_SIZE
    VAL_N_BATCH = VAL_BUFFER_SIZE // BATCH_SIZE
    ATTACKRATE_N_BATCH = ATTACKRATE_BUFFER_SIZE // BATCH_SIZE
    TEST_N_BATCH = TEST_BUFFER_SIZE // BATCH_SIZE

    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(globalVocab.word2idx)
    target_size = num_emotions

    train_dataset_clients = []

    for client in range(0,numClients):

        train_dataset = dataLoadUtils.MyData(input_tensor_clients[client], target_tensor_clients[client])
        train_dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                             drop_last=True,
                             shuffle=True)
        train_dataset_clients.append(train_dataset)



    val_dataset = dataLoadUtils.MyData(input_tensor_val, target_tensor_val)
    val_dataset = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=True)

    attackRate_dataset = dataLoadUtils.MyData(input_tensor_attackRate, target_tensor_attackRate)
    attackRate_dataset = DataLoader(attackRate_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=True)


    test_dataset = dataLoadUtils.MyData(input_tensor_test, target_tensor_test)
    test_dataset = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=True)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = emotionModel.EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
    model.to(device)

    train_dataset = train_dataset_clients

    # print(len(train_dataset))

    # # obtain one sample from the data iterator
    # it = iter(train_dataset)
    # x, y, x_len = next(it)

    # # sort the batch first to be able to use with pac_pack sequence
    # xs, ys, lens = emotionModel.sort_batch(x, y, x_len)

    # print("Input size: ", xs.size())

    # output, _ = model(xs.to(device), lens, device)
    # print(output.size())

    ### Enabling cuda
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    model = emotionModel.EmoGRU(vocab_inp_size, embedding_dim, units, BATCH_SIZE, target_size)
    model.to(device)

    ### loss criterion and optimizer for training
    criterion = nn.CrossEntropyLoss() # the same as log_softmax + NLLLoss
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.75, weight_decay=0.001)
    optimizer = torch.optim.Adam(model.parameters())

    total_loss = 0
    train_accuracy, val_accuracy = 0, 0

    # print(train_dataset)

    numIterations = 200

    datasetIter = [(iter(dataset)) for dataset in train_dataset]
    for iteration in range(0,numIterations):

        nextBatches = [next(iterator) for iterator in datasetIter]

        # print(iteration)

        aggregatedGradients = np.zeros(0)

        total_loss = 0

        prev_validation_accuracy = 0 + val_accuracy
        train_accuracy, val_accuracy, attackRate_accuracy = 0, 0, 0

        randomPeers = 0

        peerList = []

        for verifier in range(numVerifiers):
            peerList.append(random.randint(0,numClients))


        for nextBatch in nextBatches:

            inp = nextBatch[0]
            # print(inp)
            targ = nextBatch[1]
            # print(targ)
            lens = nextBatch[2]

            loss = 0
            predictions, _ = model(inp.permute(1 ,0).to(device), lens, device) # TODO:don't need
            loss += emotionModel.loss_function(targ.to(device), predictions)
            batch_loss = (loss / int(targ.shape[1]))        
            total_loss += batch_loss
            
            optimizer.zero_grad()

            loss.backward()

            modelLayers = getCurrentModel(model)

            
            layers = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    layers.append(param.grad)

            goodUpdate = roni(modelLayers, layers,prev_validation_accuracy , iteration, val_dataset)
            print(goodUpdate)
            # sys.exit()

            # goodUpdate = roni(modelLayers, layers, prev_validation_accuracy)
            # print(goodUpdate)
            # return layers
            if goodUpdate and roni:
                if len(aggregatedGradients) == 0:
                    aggregatedGradients = layers
                else:
                    for layerIdx in range(0, len(aggregatedGradients)):
                        aggregatedGradients[layerIdx] = aggregatedGradients[layerIdx] + layers[layerIdx]

        # break    
        if (len(aggregatedGradients) == 0):
            continue

        layer = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = aggregatedGradients[layer]
                layer += 1  
        
        optimizer.step()

        for (batch, (inp, targ, lens)) in enumerate(val_dataset):

            predictions,_ = model(inp.permute(1, 0).to(device), lens, device)        
            batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
            val_accuracy += batch_accuracy

        if ((val_accuracy / VAL_N_BATCH) > 90):
            print(iteration)
            break

        val_accuracy = (val_accuracy / VAL_N_BATCH)
        
        # print(' Val Acc. {:.4f}'.format(val_accuracy / VAL_N_BATCH))
            # break

        for (batch, (inp, targ, lens)) in enumerate(attackRate_dataset):

            predictions,_ = model(inp.permute(1, 0).to(device), lens, device)        
            batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
            attackRate_accuracy += batch_accuracy

        # print(' Attack Rate. {:.4f}'.format(attackRate_accuracy / ATTACKRATE_N_BATCH))
        attackRate_accuracy = attackRate_accuracy / ATTACKRATE_N_BATCH

        print('{},{:.4f},{:.4f}'.format(iteration, val_accuracy, attackRate_accuracy))


if __name__ == "__main__":
    main()

























# for batch_sets in zip([dataset for dataset in train_dataset]):

#     print(batch_sets)

    # optimizer.zero_grad()
    # aggregatedGradients = np.zeros(0)

    # for batch_set in batch_sets:

    #     for (batch, (inp, targ, lens)) in enumerate(batch_set):

    #         loss = 0
    #         predictions, _ = model(inp.permute(1 ,0).to(device), lens, device) # TODO:don't need
    #         loss += emotionModel.loss_function(targ.to(device), predictions)
    #         batch_loss = (loss / int(targ.shape[1]))        
    #         total_loss += batch_loss
            
    #         optimizer.zero_grad()
            
    #         # for p in model.parameters():
    #         #     print(p.grad)
            
    #         loss.backward()
            
    #         layers = []
    #         for name, param in model.named_parameters():
    #             if param.requires_grad:
    #                 layers.append(param.grad)
    #         # return layers

    #         if len(aggregatedGradients) == 0:
    #             aggregatedGradients = layers
    #         else:
    #             for layerIdx in range(0, len(aggregatedGradients)):
    #                 aggregatedGradients[layerIdx] = aggregatedGradients[layerIdx] + layers[layerIdx]
           
    #         # break

    #     layer = 0
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             param.grad = aggregatedGradients[layer]
    #             layer += 1  
        
    #     optimizer.step()

    #         ### Validating
    #     for (batch, (inp, targ, lens)) in enumerate(val_dataset):        
    #         predictions,_ = model(inp.permute(1, 0).to(device), lens, device)        
    #         batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
    #         val_accuracy += batch_accuracy

    #     print(' Val Acc. {:.4f}'.format(val_accuracy / VAL_N_BATCH))
    #     # break




            # break



# for (batch, (inp, targ, lens)) in enumerate(train_dataset):

#     loss = 0
#     predictions, _ = model(inp.permute(1 ,0).to(device), lens, device) # TODO:don't need 
#     loss += emotionModel.loss_function(targ.to(device), predictions)
#     batch_loss = (loss / int(targ.shape[1]))        
#     total_loss += batch_loss
    
#     optimizer.zero_grad()
#     loss.backward()
#     for p in model.parameters():
#         print(p.grad)
#     # optimizer.step()
#     # print("learned A = {}".format(list(model.parameters())[0]))
#     # print("learned b = {}".format(list(model.parameters())[1]))
#     break




# EPOCHS = 1

# for epoch in range(EPOCHS):
#     start = time.time()
    
#     ### Initialize hidden state
#     # TODO: do initialization here.
#     total_loss = 0
#     train_accuracy, val_accuracy = 0, 0
    
#     ### Training
#     for (batch, (inp, targ, lens)) in enumerate(train_dataset):
#         loss = 0
#         predictions, _ = model(inp.permute(1 ,0).to(device), lens, device) # TODO:don't need _   
              
#         loss += emotionModel.loss_function(targ.to(device), predictions)
#         batch_loss = (loss / int(targ.shape[1]))        
#         total_loss += batch_loss
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
#         train_accuracy += batch_accuracy
        
#         if batch % 100 == 0:
#             print('Epoch {} Batch {} Val. Loss {:.4f}'.format(epoch + 1,
#                                                          batch,
#                                                          batch_loss.cpu().detach().numpy()))
            
#     ### Validating
#     for (batch, (inp, targ, lens)) in enumerate(val_dataset):        
#         predictions,_ = model(inp.permute(1, 0).to(device), lens, device)        
#         batch_accuracy = emotionModel.accuracy(targ.to(device), predictions)
#         val_accuracy += batch_accuracy
    
#     print('Epoch {} Loss {:.4f} -- Train Acc. {:.4f} -- Val Acc. {:.4f}'.format(epoch + 1, 
#                                                              total_loss / TRAIN_N_BATCH, 
#                                                              train_accuracy / TRAIN_N_BATCH,
#                                                              val_accuracy / VAL_N_BATCH))
#     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))




# print(input_tensor)
# print(target_tensor)