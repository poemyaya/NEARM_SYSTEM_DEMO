import pickle
import os
import pickle,re
import numpy as np
from scipy.stats import multivariate_normal
def readpkl(url):
    with open(url,'rb') as f:
        data = pickle.load(f)
    return data
#---------------计算匹配度--------------------------
def delNum(sent):
    number = re.findall(r'(\w*[0-9]+\w*)', sent)
    for num in number:
        if num != '' and 'subjplace' not in num and 'objplace' not in num:
            if ' '+num in sent:
                sent = sent.replace(' '+num, ' numberplaceholder')
            elif num+' ' in sent:
                sent = sent.replace(num+' ', 'numberplaceholder ')
    sent = sent.strip()
    return sent
def reWordIndex(sent):
    sent = delNum(sent)
    sent = re.sub('[’!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~]+', '', sent)
    sent = sent.replace('  ', ' ')
    sent = sent.split(" ")
    ks = 0
    ko = 0
    for word in sent:
        if (ks + ko) == 2:
            break
        if 'subjplace' in word:
            subjPlace = word
            ks = 1
        if 'objplace' in word:
            objPlace = word
            ko = 1
    sindex = sent.index(subjPlace)
    oindex = sent.index(objPlace)

    # 防止一个单词出现多次，这里没有用字典
    wordIndex = []
    for i in range(len(sent)):
        wo = sent[i]
        if len(wo) != 0 and 'subjplace' not in wo and 'objplace' not in wo:
            wordIndex.append((wo, [i - sindex, i - oindex]))

    return wordIndex

def matrixSO_9(loc_s, loc_o):
    reMatrix = []
    locS = [loc_s - 0.5, loc_s + 0.5]
    locO = [loc_o - 0.5, loc_o + 0.5]

    reMatrix.append([locS[0], locO[0]])
    reMatrix.append([locS[0], locO[1]])
    reMatrix.append([locS[1], locO[0]])
    reMatrix.append([locS[1], locO[1]])

    reMatrix.append([loc_s, locO[0]])
    reMatrix.append([loc_s, locO[1]])
    reMatrix.append([locS[0], loc_o])
    reMatrix.append([locS[1], loc_o])

    reMatrix.append([loc_s, loc_o])

    return reMatrix

def func(x,meanlist,covlist):
    return multivariate_normal.pdf(x, mean=meanlist, cov=covlist)

def tryFunc(x,locList,meanlist,covlist):
    res = 0
    try:
        res = func(x,meanlist,covlist)
    except:
        randomList0 = np.random.normal(0, 0.1, len(locList[0]))
        tr_x = locList[0] + randomList0
        # randomList1 = [random.random() for i in range(len(locList[0]))]
        randomList1 = np.random.normal(0, 0.1, len(locList[0]))
        tr_y = locList[1] + randomList1
        cov = np.cov(np.vstack((tr_x, tr_y)))
        res = tryFunc(x,locList,meanlist,cov)
    return res

def reModel(wordIndex,allTemplate):
    simTemp = {}
    county = 0
    marchwordcount = {}
    for key in allTemplate.keys():
        template = allTemplate[key]
        sim = 0
        count_mw = 0
        for wordlist in wordIndex:
            word = wordlist[0]
            if word in template.keys():
                # word = word.decode("utf-8")
                count_mw += 1
                loc_s = wordlist[1][0]
                loc_o = wordlist[1][1]
                locso = []
                locso.append(float(loc_s))
                locso.append(float(loc_o))
                reMatrix = matrixSO_9(loc_s, loc_o)


                locList = template[word][0]
                meanlist = template[word][1]
                covlist = template[word][2]

                if locso == meanlist:
                    if template[word][-1] < 0.5:
                        sim = sim + 0.0001
                else:
                    y0 = tryFunc(reMatrix[0],locList,meanlist,covlist)
                    y1 = tryFunc(reMatrix[1],locList,meanlist,covlist)
                    y2 = tryFunc(reMatrix[2],locList,meanlist,covlist)
                    y3 = tryFunc(reMatrix[3],locList,meanlist,covlist)
                    y4 = tryFunc(reMatrix[4],locList,meanlist,covlist)
                    y5 = tryFunc(reMatrix[5], locList, meanlist, covlist)
                    y6 = tryFunc(reMatrix[6], locList, meanlist, covlist)
                    y7 = tryFunc(reMatrix[7], locList, meanlist, covlist)
                    y8 = tryFunc(reMatrix[8], locList, meanlist, covlist)
                    # y = ((y0+y1+y2+y3)*0.15+y4*0.4) * template[word][-1]
                    y = ((y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7) * 0.1 + y8 * 0.2) * template[word][-1]
                    if y>1:
                        y = 1
                        county += 1
                    # print(key,word,y,(y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7) * 0.1 + y8 * 0.2,template[word][-1])
                    sim = sim + y
        marchwordcount[key] = count_mw
        simTemp[key] = [sim,count_mw]
    sort_mwc = sorted(marchwordcount.items(), key=lambda x: x[1], reverse=True)
    sortSim = sorted(simTemp.items(), key=lambda x: x[1], reverse=True)
    marchvalue, marchIndex = 0,0
    rewocount = sort_mwc[0][1]
    for im2 in sortSim:
        if rewocount == im2[1][1]:
            if marchvalue<im2[1][0]:
                marchvalue, marchIndex = im2[1][0],im2[0]
    return marchvalue, marchIndex, rewocount


def reSentsMarch(sent,templateroot,template1,template2):
    reSentmatchListRoot = []
    reSentmatchDict1 = {}
    reSentmatchDict2 = {}
    wordIndex = reWordIndex(sent)
    templatedictroot = {}
    templatedictroot[0] = templateroot
    marchIndex1, marchIndex2, marchvalue1, marchvalue2, rewocount1, rewocount2 = 0,0,0,0,0,0
    marchvalueROOT,marchIndeROOT, rewocountronnt = reModel(wordIndex,templatedictroot)
    if marchvalueROOT>0:
        reSentmatchListRoot.append((sent,marchvalueROOT))
        marchvalue1, marchIndex1, rewocount1 = reModel(wordIndex,template1)
        if marchvalue1>0:
            if marchIndex1 not in reSentmatchDict1.keys():
                reSentmatchDict1[marchIndex1] = [(sent,marchvalue1)]
            else:
                reSentmatchDict1[marchIndex1].append((sent,marchvalue1))
            sencondT = template2[marchIndex1]
            marchvalue2,marchIndex2, rewocount2 = reModel(wordIndex,sencondT)
            if marchvalue2>0:
                if (marchIndex1,marchIndex2) not in reSentmatchDict2.keys():
                    reSentmatchDict2[(marchIndex1,marchIndex2)] = [(sent,marchvalue2)]
                else:
                    reSentmatchDict2[(marchIndex1,marchIndex2)].append((sent,marchvalue2))
    return marchIndex1,marchIndex2,marchvalue1,marchvalue2

def predRules(index1, index2, degree1, degree2,rules1,rules2,sent):
    ks,ko = 0,0
    for word in sent.split(' '):
        if (ks + ko) == 2:
            break
        if 'subjplace' in word:
            subjPlace = word.split('_')[0]
            ks = 1
        if 'objplace' in word:
            objPlace = word.split('_')[0]
            ko = 1
    rerules = rules2[index1][index2]
    predrules = []
    for item in rerules:
        if 'subjplace' in item[0]:
            triple = item[0].replace('subjplace',subjPlace)
        if 'objplace' in item[0]:
            triple = item[0].replace('objplace', objPlace)
        # predrules.append((triple,item[1]+degree2))
        predrules.append((triple, item[1]))
    return predrules,subjPlace,objPlace

def lasttriple(sent,attr):
    templateRoot = readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'template_DS',attr +'_1.pkl'))
    templateDicts1 =  readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'template_DS',attr +'_2.pkl'))
    templateDicts2 =  readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'template_DS',attr +'_3.pkl'))

    rulesRoot =  readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'rules_DS',attr +'_1.pkl'))
    rules1 = readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'rules_DS',attr +'_2.pkl'))
    rules2 = readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'rules_DS',attr +'_3.pkl'))
    # templateRoot = readpkl('F:/pythonworkspace/newWorkplace/Dweb/learn/template_DS/' + attr + '_1.pkl')
    # templateDicts1 = readpkl('F:/pythonworkspace/newWorkplace/Dweb/learn/template_DS/' + attr + '_2.pkl')
    # templateDicts2 = readpkl('F:/pythonworkspace/newWorkplace/Dweb/learn/template_DS/' + attr + '_3.pkl')
    #
    # rulesRoot = readpkl('F:/pythonworkspace/newWorkplace/Dweb/learn/rules_DS/' + attr + '_1.pkl')
    # rules1 = readpkl('F:/pythonworkspace/newWorkplace/Dweb/learn/rules_DS/' + attr + '_2.pkl')
    # rules2 = readpkl('F:/pythonworkspace/newWorkplace/Dweb/learn/rules_DS/' + attr + '_3.pkl')
    sent = sent.replace('_e1', '_subjplace')
    sent = sent.replace('_e2', '_objplace')

    index1, index2, degree1, degree2 = reSentsMarch \
        (sent, templateRoot, templateDicts1, templateDicts2)
    pretriples,subjPlace,objPlace = predRules(index1, index2, degree1, degree2, rules1, rules2, sent)
    if attr=='song':
        attr = 'performer'
    pretriples.append((subjPlace+'_'+attr+'_'+objPlace,''))
    npretriples = filterrules(pretriples, subjPlace, objPlace)
    # print(pretriples, len(pretriples))
    return npretriples

def relClass(attrlist,sent):

    sent = sent.replace('_e1', '_subjplace')
    sent = sent.replace('_e2', '_objplace')
    wordIndex = reWordIndex(sent)
    # print(wordIndex)
    relindex = 0
    relmarch = 0
    rewocount = 0
    for attr_i in range(len(attrlist)):
        attr = attrlist[attr_i]
        templateDicts1 = readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'template_DS',attr +'_2.pkl'))

        marchvalue1, marchIndex1, rewocount1 = reModel(wordIndex, templateDicts1)
        # print(attr,marchvalue1, marchIndex1,rewocount1)
        if rewocount1 >rewocount:
            relmarch = marchvalue1
            relindex = attr_i
            rewocount = rewocount1
        elif rewocount1 == rewocount:
            if marchvalue1>relmarch:
                relmarch = marchvalue1
                relindex = attr_i
                rewocount = rewocount1
        templateDicts2 = readpkl(os.path.join(os.path.dirname(os.path.realpath(__file__)),'template_DS',attr +'_3.pkl'))
        # templateDicts2 = readpkl('F:/pythonworkspace/newWorkplace/Dweb/learn/template_DS/' + attr + '_3.pkl')
        sencondT = templateDicts2[marchIndex1]
        marchvalue2, marchIndex2, rewocount2 = reModel(wordIndex, sencondT)
        if rewocount2 > rewocount:
            relmarch = marchvalue2
            relindex = attr_i
            rewocount = rewocount2
        elif rewocount2 == rewocount:
            if marchvalue2 > relmarch:
                relmarch = marchvalue2
                relindex = attr_i
                rewocount = rewocount2
        # print(attr,marchvalue2, marchIndex2,rewocount2)
    # print(relmarch, relindex,attrlist[relindex])
    return attrlist[relindex]


def filterrules(newr,subjPlace,objPlace):
    filtlist = [subjPlace+'_gender',objPlace+'_gender']
    re_newrules = []
    attrrulelist = []
    for item in newr:
        rule = item[0]
        #过滤diao置信度小于阈值的三元组
        if item[1] !='':
            if item[1]<0.5:
                continue
        split_i = rule.split('_')
        attrrule = split_i[0]+'_'+split_i[1]
        if attrrule in filtlist:
            if attrrule not in attrrulelist:
                re_newrules.append(item)
                attrrulelist.append(attrrule)
            else:
                for i_item in re_newrules:
                    if attrrule in i_item[0]:
                        if item[1]>i_item[1]:
                            re_newrules.remove(i_item)
                            re_newrules.append(item)
        else:
            re_newrules.append(item)
    return re_newrules

def modelinter(sent):
    attrlist = ['song', 'spouse', 'mother', 'film', 'father', 'child', 'filmCastmember', 'songPerformer']
    # if 'is the daughter of' in sent:
    #     attr1 = 'father'
    # else:
    #     attr1 = relClass(attrlist, sent)
    if '_e1' not in sent or '_e2' not in sent:
        return None
    attr1 = relClass(attrlist, sent)
    triples = lasttriple(sent, attr1)
    return triples

cwd = os.getcwd()
static_dir = os.path.join(cwd, 'static')
# 三元组的保存路径
triple_dir = os.path.join(static_dir, 'triple')
path = os.path.join(triple_dir, 'org.pickle')


class Mod:
    def __init__(self):
        self.mod = {}
        s1 = 'Only@Sixteen_e1 is a song by American singer-songwriter Sam@Cooke_e2'
        s2 = 'Womack_e1 is the daughter of Sam@Cooke_e2'
        triple1 = [
            ('Sam@Cooke_instanceof_human', 0.93887147335423193),
            ('Sam@Cooke_occupation_songwriter', 0.54545454545454541),
            ('Sam@Cooke_occupation_composer', 0.6630094043887147),
            ('Sam@Cooke_occupation_singer', 0.41692789968652039),
            ('Only@Sixteen_instanceof_song', 0.92319749216300939),
            ('Sam@Cooke_gender_male', 0.84012539184952983),
            ('Sam@Cooke_citizenship_United States of America', 0.56896551724137934)
        ]
        triple2 = [
            ('Sam@Cooke_instanceof_human', 0.95272206303724927),
            ('Womack_gender_female', 0.80229226361031514),
            ('Womack_instanceof_human', 0.9240687679083095),
            ('Sam@Cooke_gender_male', 0.95272206303724927)
        ]
        self.mod[s1] = triple1
        self.mod[s2] = triple2


def delfile(dels):
    if dels == 'true' and os.path.exists(path):
        print(path)
        os.remove(path)

def runfile(runs):
    if runs == 'true' and os.path.exists(path):
        f = open(path, 'rb')
        data = pickle.load(f)
        nodes = data['nodes']
        rels = data['relations']
        node_name = data['node_name']
        triple_set = data['triple_set']
        f.close()
    else:
        nodes = []
        rels = []
        node_name = set()
        triple_set = set()
    return nodes, rels

def get_triples(sent):
    # 已保存的三元组
    if os.path.exists(path):
        f = open(path, 'rb')
        data = pickle.load(f)
        nodes = data['nodes']
        rels = data['relations']
        node_name = data['node_name']
        triple_set = data['triple_set']
        f.close()
    else:
        nodes = []
        rels = []
        node_name = set()
        triple_set = set()
    sent = sent.strip()
    # 输入Empty Existing Files时删除已保存的三元组
    # if dels == 'true' and os.path.exists(path):
    #     os.remove(path)
    # M = Mod()
    # mod = M.mod
    # triples = mod.get(sent)
    triples = modelinter(sent)



    # 添加输入句子的三元组
    if triples is None:
        return nodes, rels
    for temp in triples:
        triple = temp[0]
        t = triple.split('_')
        conf = temp[1]
        if conf == '':
            conf = 0
        else:
            conf = float(conf)

        conf = round(conf, 3)
        sub = {'name': t[0].replace('@',' ')}
        obj = {'name': t[-1].replace('@',' ')}
        rel = {'name': t[1], 'conf': conf, 'source': sub.get('name'), 'target': obj.get('name')}
        if rel['conf']==0:
            rel['symbolSize']='1'
        if sub['name'] not in node_name:
            node_name.add(sub['name'])
            nodes.append(sub)
        if obj['name'] not in node_name:
            node_name.add(obj['name'])
            nodes.append(obj)
        if triple not in triple_set:
            triple_set.add(triple)
            rels.append(rel)

    # 保存已输入句子的三元组
    if not os.path.exists(triple_dir):
        os.makedirs(triple_dir)
    f1 = open(path, mode='wb')
    data = {'nodes': nodes, 'relations': rels, 'node_name': node_name, 'triple_set': triple_set}
    pickle.dump(data, f1)
    f1.close()
    # print(nodes)
    # print(rels)
    return nodes, rels


def fun():
    sub = {'name': 'Tom', 'label': 'cat'}
    obj = {'name': 'Jerry', 'label': 'mouse'}
    nodes = [sub, obj]
    rel = {'name': 'friend', 'source': sub.get('name'), 'target': obj.get('name')}
    rels = [rel]
    return nodes, rels


if __name__ == '__main__':
    print(path)
    print(modelinter('Only@Sixteen_e1 is a song by American singer-songwriter Sam@Cooke_e2'))
    print(modelinter('Womack_e1 is the daughter of Sam@Cooke_e2'))
