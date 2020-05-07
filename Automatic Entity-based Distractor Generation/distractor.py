import json
import unicodedata
import argparse
import numpy as np
import spacy
import pywikibot
import requests
import collections
from useBERT import Vocab
from feature import Feature, getEntityBERTEmb, getGoldBERTEmb, getQuesBERTEmb
import time



class Article:
    def __init__(self, title, contextAll):
        self.title = title
        self.contextAll = contextAll
        self.QAList = []
        self.entityList = []
        self.entityDict = {}

    def nlp(self):
        '''
        named entity recognition and pos tagging for a give article

        '''
        '''
        spaCy NER labels
        TYPE	    DESCRIPTION
        PERSON	    People, including fictional.
        NORP	    Nationalities or religious or political groups.
        FAC	        Buildings, airports, highways, bridges, etc.
        ORG	        Companies, agencies, institutions, etc.
        GPE	        Countries, cities, states.
        LOC	        Non-GPE locations, mountain ranges, bodies of water.
        PRODUCT	    Objects, vehicles, foods, etc. (Not services.)
        EVENT	    Named hurricanes, battles, wars, sports events, etc.
        WORK_OF_ART	Titles of books, songs, etc.
        LAW	Named   documents made into laws.
        LANGUAGE	Any named language.
        DATE	    Absolute or relative dates or periods.
        TIME	    Times smaller than a day.
        PERCENT	    Percentage, including ”%“.
        MONEY	    Monetary values, including unit.
        QUANTITY	Measurements, as of weight or distance.
        ORDINAL	    “first”, “second”, etc.
        CARDINAL	Numerals that do not fall under another type.
        '''
        doc = nlp(self.contextAll)
        # pos tagging
        self.posDict = {}
        for token in doc:
            self.posDict[token.idx] = (token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                                       token.shape_, token.is_alpha, token.is_stop)
        # named entity recognition
        self.entityList = doc.ents
        self.entitySet = set()  # remove duplicate entity with same text and different index
        self.entityUniqueList = []  # set of tuple (entity text, entity start id), which is all possible distractors list
        self.entityDict = collections.defaultdict(
            set)  # key: entity label, val: set of tuple (entity text, entity start id)
        self.entityText = collections.defaultdict(str)  # key: entity text, val: entity label
        self.entityStartIdx = collections.defaultdict(
            str)  # key: entity start index (char level, inclusive), val: entity label
        self.entityBERTEmb = {} # key: entity text, val: BERT embedding
        for entity in self.entityList:
            if entity.text not in self.entitySet:
                self.entityDict[entity.label_].add((entity.text, entity.start_char))
                self.entityText[entity.text] = entity.label_
                self.entityStartIdx[entity.start_char] = entity.label_
                self.entitySet.add(entity.text)
                self.entityUniqueList.append((entity.text, entity.start_char))
                # ent.start_char, ent.end_char

        print(self.entityDict)


class QA:
    def __init__(self, question, isImpossible):
        self.question = question
        self.questionCoarseType = None
        self.questionFineType = None
        self.gold = None
        self.goldStartIdx = None
        self.goldEndIdx = None
        self.goldNERType = None
        self.goldDescription = None
        self.isImpossible = isImpossible
        self.distractorCandidates = set()


def pyWikiBot(query):
    site = pywikibot.Site("en", "wikipedia")
    page = pywikibot.Page(site, query)
    item = pywikibot.ItemPage.fromPage(page)
    return item


def wbSearchEntities(query):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': query
    }
    API_ENDPOINT = "https://www.wikidata.org/w/api.php"
    r = requests.get(API_ENDPOINT, params=params)
    return [i['description'] if 'description' in i else None for i in r.json()['search']]


def questionClassificationAPI(question, authcode):
    '''
    Coarse	Fine
    ABBR	abbreviation; expansion
    DESC	definition; description; manner; reason
    ENTY	animal; body; color; creation; currency; disease; event; food;
            instrument; language; letter; other; plant; product; religion;
            sport; substance; symbol; technique; term; vehicle; word; extraterrestrial;
    HUM	    description; group; individual; title; indgr
    LOC	    city; country; mountain; other; state
    NUM	    code; count; date; distance; money; order; other; percent; period;
            speed; temperature; size; weight

    Note:
    1. the fine class “extraterrestrial” to capture entities such as “planet”, “moon” and so on
    2. the fine class “indgr” to represent questions that cannot be classified into either the class individual
       or the class group without knowledge of the answer (an example of this is “Who won the Nobel Peace Prize
       in <year>?”)
    3. the reasoning behind the classification to classify human made entities such as “bridge” and “building”
       under ENTY:other instead of the previous classification of LOC:other.

    API response example
    {
    "status": "Success",
    "major_type": "LOC",
    "minor_type": "city",
    "question_text": "What is the capital of the UK?",
    "syntactic_map": "
         wh_word : ('What', (0, 4))
         WHNP :
            NNP : None
         SQ :
            auxiliary_verb : ['is']
            NNP            : capital(PP) of PP-NN: UK
            main_verb      : None
    "
    }
    '''

    def strip_accents(text):
        return ''.join(char for char in unicodedata.normalize('NFKD', text)
                       if unicodedata.category(char) != 'Mn')

    authcode = authcode
    url = 'http://qcapi.harishmadabushi.com/?auth=' + authcode \
          + '&question=' + strip_accents(question).replace(' ', '%20')
    r = requests.get(url)
    try:
        # print(r.json()['status'])
        return r.json()
    except:
        return None


# print(questionClassificationAPI('When did Beyonce start becoming popular?'))


def getDistractorCandidatesNERTypeSet(questionCoarseType, questionFineType, goldNERType):
    # mapping table from api tags to spaCy tags
    res = set()
    cond = 0  # 0 for neither, 1 for only gold, 2 for only question, 3 for both
    if goldNERType:
        cond = 1
        res.add(goldNERType)
    if questionCoarseType:
        if cond == 0:
            cond = 2
        else:
            cond = 3
        if questionCoarseType == 'HUM':
            res.add('PERSON')
        elif questionCoarseType == 'LOC':
            res.update(['GPE', 'LOC'])

        elif questionCoarseType == 'NUM':
            if questionFineType in ['date', 'year', 'period']:
                res.add('DATE')
            elif questionFineType == 'percent':
                res.add('PERCENT')
            elif questionFineType == 'money':
                res.add('MONEY')
            elif questionFineType in ['dist', 'speed', 'temp', 'size', 'weight']:
                res.add('QUANTITY')
            elif questionFineType == 'order':
                res.add('ORDINAL')
            elif questionFineType in ['count', 'volsize']:
                res.add('CARDINAL')
            elif questionFineType == 'other':
                res.update(['MONEY', 'CARDINAL'])
        elif questionCoarseType == 'ENTY':
            if questionFineType == 'lang':
                res.update(['NORP', 'LANGUAGE'])
            elif questionFineType == 'cremat':
                res.add('WORK_OF_ART')
            elif questionFineType == 'religion':
                res.add('NORP')
            elif questionFineType in ['event', 'sport']:
                res.add('EVENT')
            elif questionFineType == 'other':
                res.add('ORG')
            else:
                res.update(['FAC', 'PRODUCT', 'LAW'])
    return res, cond


def processSQuADtrain(trainfile, destfile, useQuestionClassificationAPI, authcode):
    '''
    Generate "Question	Gold	Top3_Distractors	Q_Coarse	Q_Fine	Gold_spaCy	Candidate_Type" for SQuAD train dataset
    :param trainfile: SQuAD train dataset 'SQuAD/train-v2.0.json'
    :param destfile: destination file, saving "Question	Gold	Top3_Distractors	Q_Coarse	Q_Fine	Gold_spaCy	Candidate_Type" columns as a tsv file
    :param useQuestionClassificationAPI: true/false, for detail please refer to http://www.harishmadabushi.com/research/questionclassification/question-classification-api-documentation/
    :param authcode: authcode for QuestionClassificationAPI, please contact http://www.harishmadabushi.com/research/questionclassification/question-classification-api-documentation/
    :return: None
    '''
    with open(trainfile) as f:
        data = json.load(f)
    with open(destfile, 'w') as fw:
        fw.write(
            '\t'.join(['Question', 'Gold', 'Top3_Distractors', 'Q_Coarse', 'Q_Fine', 'Gold_spaCy', 'Candidate_Type']) + '\n')
        articleList = []
        for i, article in enumerate(data['data']):  # SQuAD train dataset num of articles = 442
            title = article['title']
            paragraphs = article['paragraphs']
            contextAll = ''
            QAList = []

            for paragraph in paragraphs:
                contextAll += paragraph['context'] + '\n'  # merge all paragraphs
            article = Article(title=title, contextAll=contextAll)
            article.nlp()  # named entity recognition
            article.entityBERTEmb = getEntityBERTEmb(article.entitySet)
            for paragraph in paragraphs:
                for qid, qa in enumerate(paragraph['qas']):

                    curQA = QA(question=qa['question'], isImpossible=qa['is_impossible'])
                    response = None
                    if useQuestionClassificationAPI.lower() == 'true':
                        response = questionClassificationAPI(curQA.question, authcode)
                    if response and response['status'] == 'Success':
                        curQA.questionCoarseType = response['major_type']
                        curQA.questionFineType = response['minor_type']
                    if curQA.isImpossible:
                        curQA.gold = None
                        curQA.goldStartIdx = None
                        curQA.goldEndIdx = None
                        curQA.goldNERType = None
                    else:
                        curQA.gold = qa['answers'][0]['text']
                        curQA.goldStartIdx = qa['answers'][0]['answer_start']
                        curQA.goldEndIdx = curQA.goldStartIdx + len(curQA.gold)
                        if curQA.gold in article.entityText:  # if gold exactly matches an entity
                            curQA.goldNERType = article.entityText[curQA.gold]
                        else:
                            curQA.goldNERType = None
                            for start in range(curQA.goldStartIdx,
                                               curQA.goldEndIdx):  # if gold contains part of an entity
                                if start in article.entityStartIdx:
                                    curQA.goldNERType = article.entityStartIdx[start]
                                    break
                    distractorCandidatesNERTypeSet, condition = getDistractorCandidatesNERTypeSet(
                        curQA.questionCoarseType,
                        curQA.questionFineType,
                        curQA.goldNERType)
                    for type in distractorCandidatesNERTypeSet:
                        curQA.distractorCandidates.update(article.entityDict[type])

                    QAList.append(curQA)

                    print('P' + str(i + 1) + 'Q' + str(qid + 1) + ': ' + curQA.question)
                    fw.write(curQA.question + '\t')
                    print('Gold:', curQA.gold)
                    fw.write((curQA.gold if curQA.gold else 'None') + '\t')

                    # print(curQA.distractorCandidates)
                    # t3 = time.time()
                    finallist = []
                    goldBERTEmb = getGoldBERTEmb(curQA.gold)
                    quesBERTEmb = getQuesBERTEmb(curQA.question)
                    for d in curQA.distractorCandidates:
                        if curQA.gold and (curQA.gold.lower() in d[0].lower() or d[0].lower() in curQA.gold.lower()):
                            continue
                        feature = Feature(embed, vocab, curQA.question, curQA.gold, d[0], curQA.goldStartIdx, d[1],
                                          article.posDict, article.entityBERTEmb, goldBERTEmb, quesBERTEmb)

                        score = feature.score
                        finallist.append([score, d[0]])
                    # t4 = time.time()
                    finallist.sort(reverse=True)
                    print('My candidate:', [c[1] for c in finallist[:3]])
                    fw.write(str([c[1] for c in finallist[:3]]) + '\t')
                    print('Ques API tag:', curQA.questionCoarseType, ',', curQA.questionFineType)
                    fw.write((curQA.questionCoarseType if curQA.questionCoarseType else 'None') + '\t' + \
                             (curQA.questionFineType if curQA.questionFineType else 'None') + '\t')
                    print('Gold spaCy tag:', curQA.goldNERType)
                    fw.write((curQA.goldNERType if curQA.goldNERType else 'None') + '\t')
                    print('distractorCandidatesNERTypeSet', distractorCandidatesNERTypeSet)
                    fw.write(str(distractorCandidatesNERTypeSet) + '\t\n')
                    print('')
                    # print('t2-t1, t4-t3', t2-t1, t4-t3)
                    # print('WikiData Description:', wbSearchEntities(curQA.gold))

            article.QAList = QAList
            articleList.append(article)
            # visualize
            # displacy.serve(nlp(contextAll), style="ent")

            # if i == 0:
            #     break

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--useQuestionClassificationAPI', help='true/false, for detail please refer to http://www.harishmadabushi.com/research/questionclassification/question-classification-api-documentation/')
    parser.add_argument('--authcode', help='authcode for QuestionClassificationAPI, please contact http://www.harishmadabushi.com/research/questionclassification/question-classification-api-documentation/')
    args = parser.parse_args()
    print(args.useQuestionClassificationAPI)
    print(args.authcode)
    # spaCy related: to download pretrained model: python -m spacy download en_core_web_lg
    nlp = spacy.load('en_core_web_lg')
    embed = np.loadtxt('BERT/bert_embed.txt')  # 19616 * 768
    vocab = Vocab('BERT/vocab20000', 20000)

    processSQuADtrain('SQuAD/train-v2.0.json', 'destination.txt', args.useQuestionClassificationAPI, args.authcode)
