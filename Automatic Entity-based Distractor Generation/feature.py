import nltk
from nltk.metrics import edit_distance
import numpy as np
from bert_serving.client import BertClient

bc = BertClient()


def getEntityBERTEmb(entitySet):
    entityBERTEmb = {}
    for entity in entitySet:
        entity = entity.strip().lower()
        entityBERTEmb[entity] = bc.encode([entity])[0]
    return entityBERTEmb

def getGoldBERTEmb(gold):
    if not gold:
        return None
    gold = gold.strip().lower()
    goldBERTEmb = bc.encode([gold])[0]
    print(goldBERTEmb.shape)
    return goldBERTEmb

def getQuesBERTEmb(question):
    question = question.strip().lower()
    quesBERTEmb = bc.encode([question])[0]

    return quesBERTEmb

class Feature:
    def __init__(self, embed, vocab, question, answer, distractor, answerIdx, distractorIdx, posDict, entityBERTEmb, goldBERTEmb, quesBERTEmb):
        self.question = question.strip().lower()
        self.answer = answer.strip().lower() if answer else None
        self.questionList = nltk.word_tokenize(self.question)
        self.answerList = nltk.word_tokenize(self.answer) if answer else []
        self.answerIdx = answerIdx if answer else -1
        self.distractor = distractor.strip().lower()
        self.distractorList = nltk.word_tokenize(self.distractor)
        self.distractorIdx = distractorIdx

        ''' BERT Embedding '''
        self.entityBERTEmb = entityBERTEmb
        self.goldBERTEmb = goldBERTEmb
        self.quesBERTEmb = quesBERTEmb
        self.embed = embed
        self.vocab = vocab
        # Embedding similarity between question(q) and distractor(d)
        # self.embSim_qd = self.QembCosineSimilarity(self.question, self.distractor)
        self.embSim_qd = self.embConsineSimlarity(self.questionList, self.distractorList)
        # Embedding similarity between answer(a) and distractor(d)
        # self.embSim_ad = self.AembCosineSimilarity(self.answer, self.distractor) if answer else 0
        self.embSim_ad = self.embConsineSimlarity(self.answerList, self.distractorList) if answer else 0
        ''' POS tagging '''
        self.posDict = posDict
        # Jaccard similarity between a and d’s simple POS tags and detailed POS tags
        self.simplePosSim, self.detailedPosSim = self.posTagSimilarity(self.answer, self.answerIdx, self.distractor,
                                                                       self.distractorIdx) if answer else (0, 0)
        ''' Edit Distance '''
        # Edit distance between a and d
        self.editDist = edit_distance(self.answer, self.distractor) if answer else 0

        ''' Token Similarity '''
        # Jaccard similarities between a and d’s tokens
        self.tokenSim = self.tokenSimilarity(self.answerList, self.distractorList) if answer else 0

        ''' Character Length Difference '''
        self.lengthDiff = abs(len(self.answer) - len(self.distractor)) if answer else 0
        self.score = self.embSim_qd + self.embSim_ad + self.simplePosSim + self.detailedPosSim + (
                1.0 / (self.editDist + 1)) + self.tokenSim + 1.0 / (self.lengthDiff + 1)

    def sentEmb(self, sentList):
        '''
        average of word embeddings
        '''
        sentEmb = np.zeros(768)
        for word in sentList:
            # if self.vocab.word2id(word) == 0:
            # print('warning OOV', word)
            sentEmb += self.embed[self.vocab.word2id(word)]
        sentEmb /= len(sentList)
        return sentEmb

    def embConsineSimlarity(self, sentList1, sentList2):
        sentEmb1 = self.sentEmb(sentList1)
        sentEmb2 = self.sentEmb(sentList2)
        cosineSimilarity = np.inner(sentEmb1, sentEmb2) / (np.linalg.norm(sentEmb1) * np.linalg.norm(sentEmb2))
        return cosineSimilarity

    def QembCosineSimilarity(self, ques, distractor):
        '''
        compute BERT embedding cosine similarity for two sentences (phrases)
        '''
        distractorEmb = self.entityBERTEmb[distractor]
        quesEmb = self.quesBERTEmb

        cosineSimilarity = np.inner(quesEmb, distractorEmb) / (np.linalg.norm(quesEmb) * np.linalg.norm(distractorEmb))
        return cosineSimilarity

    def AembCosineSimilarity(self, gold, distractor):
        '''
        compute BERT embedding cosine similarity for two sentences (phrases)
        '''
        distractorEmb = self.entityBERTEmb[distractor]
        goldEmb = self.goldBERTEmb

        cosineSimilarity = np.inner(goldEmb, distractorEmb) / (np.linalg.norm(goldEmb) * np.linalg.norm(distractorEmb))
        return cosineSimilarity

    def posTagSet(self, sent, startId):
        endId = startId + len(sent)
        simplePosTagSet = set()
        detailedPosTagSet = set()
        for i in range(startId, endId):
            if i in self.posDict:
                simplePosTagSet.add(self.posDict[i][2])  # The simple part-of-speech tag.
                detailedPosTagSet.add(self.posDict[i][3])  # The detailed part-of-speech tag.
        return simplePosTagSet, detailedPosTagSet

    def jaccardSimilarity(self, s1, s2):
        '''
        Jaccard similarity between two sets
        J(X,Y) = |X∩Y| / |X∪Y|
        '''
        return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))

    def posTagSimilarity(self, sent1, idx1, sent2, idx2):
        simpleSet1, detailedSet1 = self.posTagSet(sent1, idx1)
        simpleSet2, detailedSet2 = self.posTagSet(sent2, idx2)
        simpleSim = self.jaccardSimilarity(simpleSet1, simpleSet2)
        detailedSim = self.jaccardSimilarity(detailedSet1, detailedSet2)
        return simpleSim, detailedSim

    def tokenSimilarity(self, sent1List, sent2List):
        tokenSet1 = set(sent1List)
        tokenSet2 = set(sent2List)
        return self.jaccardSimilarity(tokenSet1, tokenSet2)
