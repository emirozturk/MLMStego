from transformers import BertForMaskedLM, AutoTokenizer
import torch
import math
from tqdm import tqdm
import string 


def removePunctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    clean_string = input_string.translate(translator)
    return clean_string


def getUnique(nonuniquelist):
    newList = []
    for l in nonuniquelist:
        if l not in newList:
            newList.append(l)
    return newList

def torchEncode(secret,bertTokenizer, bertModel,predSize,device):
    secret = removePunctuation(secret)
    secret = secret.replace("MASK","[MASK]")
    predCount = predSize
    ids = bertTokenizer(secret,padding=True, truncation=True,add_special_tokens=True, return_tensors='pt')
    mask_token_index = [torch.where(x==bertTokenizer.mask_token_id)[0] for x in ids["input_ids"]]
    ids = ids.to(device)
    predict = bertModel(**ids)[0]

    predictionList = []
    for i in range(len(predict)): 
        predictionList.append(torch.topk(predict[i, mask_token_index[i], :],predCount,dim=1)[1])

    resultList = []
    for pred in predictionList:
        result = ""
        if len(pred)>0:
            result = [x.replace("##","") for x in bertTokenizer.convert_ids_to_tokens(pred[0])]
        resultList.append(result)

    resultList = getUnique(resultList[0])
    return resultList


def torchInit(bertModelName,device):    
    bert_tokenizer = AutoTokenizer.from_pretrained(bertModelName,model_max_length=50)
    bert_model = BertForMaskedLM.from_pretrained(bertModelName)
    bert_model.eval()
    bert_model.to(device)
    return bert_tokenizer, bert_model


def intToPackets(intValue,packetSize):
    value = bin(intValue)[2:]
    value = value.zfill(32)
    pad = (packetSize - len(value)%packetSize)%packetSize
    value = value.zfill(len(value)+pad)
    packets=[]
    for i in range(0,len(value),packetSize):
        x = value[i:i+packetSize]
        i = int(x,2)
        packets.append(i)
    return packets


def packetsToInt(packets,packetsize):
    binValue = ""
    for packet in packets:
        binValue += str(bin(packet)[2:]).zfill(packetsize)
    value = int(binValue,2)
    return value


def charsToInt(charArray):
    byteArray = bytes(charArray)
    intValue = int.from_bytes(byteArray, byteorder='big')
    return intValue


def intToChars(intValue):
    byteArray = intValue.to_bytes(4, byteorder='big')
    charArray = [chr(byte) for byte in byteArray]
    return charArray


def removeAtIndex(input_string, index):
    if index < 0 or index >= len(input_string):
        return input_string
    return input_string[:index] + input_string[index+1:]


def squareHash(seed,value,maxMod):
    value = str(value)
    value = int(value[(len(value)-1)//2:(len(value)+2)//2])
    value = value*value
    value = (value+seed) % maxMod
    if value == 0: value =seed % maxMod
    return value


def getRandomWordIndex(randomSeed,emptyIndexes,lastIndex):
    val = squareHash(randomSeed,lastIndex,len(emptyIndexes))
    index = emptyIndexes.pop(val)
    return index,emptyIndexes


def getEmptyIndexes(totalWordCount,start,halfWindowSize):
    return list(range(start+halfWindowSize,totalWordCount,halfWindowSize*2))


def encode(secret,coverText,randomSeed,halfWindowSize,loopChange,loopMod,bertTokenizer,bertModel,device):
    wordCount = len(secret)
    originalWords = coverText.split(" ")

    predSize = 2**7
    seedPackets = intToPackets(randomSeed,7)
    wordCountPackets = intToPackets(wordCount,7)
    loopChangePackets = intToPackets(loopChange,7)

    index = 1
    for byte in seedPackets:
        originalWords[index]="[MASK]"
        predictions = torchEncode(" ".join(originalWords[index:index+halfWindowSize]),bertTokenizer,bertModel,predSize,device)
        originalWords[index] =predictions[byte]
        index+=halfWindowSize

    for byte in wordCountPackets:
        originalWords[index]="[MASK]"
        predictions = torchEncode(" ".join(originalWords[index:index+halfWindowSize]),bertTokenizer,bertModel,predSize,device)
        originalWords[index] =predictions[byte]
        index+=halfWindowSize

    for byte in loopChangePackets:
        originalWords[index]="[MASK]"
        predictions = torchEncode(" ".join(originalWords[index:index+halfWindowSize]),bertTokenizer,bertModel,predSize,device)
        originalWords[index] =predictions[byte]
        index+=halfWindowSize

    loopValue=0
    counter=0
    emptyIndexes = getEmptyIndexes(len(originalWords),index,halfWindowSize)
    if len(emptyIndexes)<wordCount:
        raise Exception("Insufficient cover text length...")

    print(emptyIndexes)
    with tqdm(total=wordCount) as pbar:
        while counter < wordCount:
            index,emptyIndexes = getRandomWordIndex(randomSeed,emptyIndexes,index)
            originalWords[index]="[MASK]"
            predictions = torchEncode(" ".join(originalWords[index-halfWindowSize:index+halfWindowSize]),bertTokenizer,bertModel,257,device)
            escape=predictions[0]
            predictions = [x for x in predictions[1:]]
            foundWords = [s for s in predictions if len(s) > loopValue and s[loopValue] == secret[counter]]
            if len(foundWords)>0:
                originalWords[index] = foundWords[0]
                counter+=1
                pbar.update(1)
            else:
                originalWords[index] = escape
            loopValue = (loopValue + loopChange) % loopMod

    outputText = " ".join(originalWords)
    return outputText


def decode(stegoMedia,halfWindowSize,loopMod,bertTokenizer,bertModel,device):
    words = stegoMedia.split(" ")
    index = 1
    loopValue = 0

    predSize = 2**7

    packetByteCount = math.ceil(32/7)
    randomSeedPackets = []
    while len(randomSeedPackets)<packetByteCount:
        word = words[index]
        words[index] = "[MASK]"
        predictions = torchEncode(" ".join(words[index:index+halfWindowSize]),bertTokenizer,bertModel,predSize,device)
        predIndex = predictions.index(word)
        randomSeedPackets.append(predIndex)
        index+=halfWindowSize

    wordCountPackets = []
    while len(wordCountPackets)<packetByteCount:
        word = words[index]
        words[index] ="[MASK]"
        predictions = torchEncode(" ".join(words[index:index+halfWindowSize]),bertTokenizer,bertModel,predSize,device)
        predIndex = predictions.index(word)
        wordCountPackets.append(predIndex)
        index+=halfWindowSize

    loopChangePackets = []
    while len(loopChangePackets)<packetByteCount:
        word = words[index]
        words[index] ="[MASK]"
        predictions = torchEncode(" ".join(words[index:index+halfWindowSize]),bertTokenizer,bertModel,predSize,device)
        predIndex = predictions.index(word)
        loopChangePackets.append(predIndex)
        index+=halfWindowSize

    randomSeed = packetsToInt(randomSeedPackets,7)
    wordCount = packetsToInt(wordCountPackets,7)
    loopChange = packetsToInt(loopChangePackets,7)


    outputText = []

    counter = 0
    emptyIndexes = getEmptyIndexes(len(words),index,halfWindowSize)
    with tqdm(total=wordCount) as pbar:
        while counter < wordCount:
            index,emptyIndexes = getRandomWordIndex(randomSeed,emptyIndexes,index)
            word = words[index]
            words[index] ="[MASK]"
            predictions = torchEncode(" ".join(words[index-halfWindowSize:index+halfWindowSize]),bertTokenizer,bertModel,257,device)
            if word != predictions[0]:
                outputText += word[loopValue]
                counter+=1
                pbar.update(1)
            loopValue= (loopValue + loopChange) % loopMod
    return "".join(outputText)

