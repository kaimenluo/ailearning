import numpy as np
import pulp
import timeit
from gensim.parsing.preprocessing import remove_stopwords

# 读取Glove文件。 注意： 不要试图修改文件以及路径
glovefile = open("glove.6B.100d.txt", "r", encoding="utf-8")

embeddings_index = {}
for line in glovefile:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
glovefile.close()

# print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 100
def get_embedding_matrix(word):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        return embedding_vector[:embedding_dim]
    return np.zeros(embedding_dim)

def tokens_to_fracdict(tokens):

    set_tokens = set(tokens)
    dict_cnt = {l:0 for l in set_tokens}
    for token in tokens:
        dict_cnt[token] += 1
    total_cnt = len(tokens)
    return {token: float(count) / total_cnt for token,count in dict_cnt.items()}

def calc_euclidean(vector1, vector2):
    return np.sqrt(sum([(item1 - item2)**2 for item1, item2 in zip(vector1, vector2)]))
    # return np.sqrt(np.power(vector1 - vector2, 2).sum())

def create_word_relations(first_sent_tokens, second_sent_tokens):
    list1 = [(item1, item2) for item1 in first_sent_tokens for item2 in second_sent_tokens]
    list2 = [(item2, item1) for item2 in second_sent_tokens for item1 in first_sent_tokens]
    return list1+list2

def removestop_and_split(sentence):
    return remove_stopwords(sentence).split()

# TODO: 编写WMD函数来计算两个句子之间的相似度
def WMD(sent1, sent2):
    """
    这是主要的函数模块。参数sent1是第一个句子， 参数sent2是第二个句子，可以认为没有经过分词。
    在英文里，用空格作为分词符号。

    在实现WMD算法的时候，需要用到LP Solver用来解决Transportation proboem. 请使用http://cvxopt.org/examples/tutorial/lp.html
    也可以参考blog： https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html

    需要做的事情为：

    1. 对句子做分词： 调用 .split() 函数即可
    2. 获取每个单词的词向量。这需要读取文件之后构建embedding matrix.
    3. 构建lp问题，并用solver解决

    可以自行定义其他的函数，但务必不要改写WMD函数名。测试时保证WMD函数能够正确运行。
    """
    first_sent_tokens = removestop_and_split(sent1)
    second_sent_tokens = removestop_and_split(sent2)

    # sent_1_embedding_matrix = get_embedding_matrix(sent1_list)
    # sent_2_embedding_matrix = get_embedding_matrix(sent2_list)

    # print(sent_1_embedding_matrix)
    # print(sent_2_embedding_matrix)

    all_tokens = list(set(first_sent_tokens + second_sent_tokens))
    wordvecs = {token: get_embedding_matrix(token) for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
    second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

    list_relations = create_word_relations(first_sent_tokens, second_sent_tokens)

    T = pulp.LpVariable.dicts('T_matrix', list(list_relations), lowBound=0)

    prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2] * calc_euclidean(wordvecs[token1], wordvecs[token2])
                        for token1, token2 in list_relations])
    for token2 in second_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets]) == second_sent_buckets[token2]
    for token1 in first_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets]) == first_sent_buckets[token1]

    prob.solve()

    wmd_dist = pulp.value(prob.objective)
    return wmd_dist


def WCD(sent1, sent2):

    first_sent_tokens = removestop_and_split(sent1)
    second_sent_tokens = removestop_and_split(sent2)

    all_tokens = list(set(first_sent_tokens + second_sent_tokens))
    wordvecs = {token: get_embedding_matrix(token) for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
    second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

    xd1 = np.zeros(100)
    xd2 = np.zeros(100)

    for token, prob in first_sent_buckets.items():
        xd1 += wordvecs[token] * prob

    for token, prob in second_sent_buckets.items():
        xd2 += wordvecs[token] * prob

    return np.sqrt(sum((xd1 - xd2)**2))


def RWMD(sent1, sent2):

    first_sent_tokens = removestop_and_split(sent1)
    second_sent_tokens = removestop_and_split(sent2)

    D1 = tokens_to_fracdict(first_sent_tokens)
    D2 = tokens_to_fracdict(second_sent_tokens)

    L1 = len(D1.items())
    L2 = len(D2.items())  # length of nbow 2
    LR1 = 0  # Relaxed solution 1
    LR2 = 0  # Relaxed solution 2

    # Calculate the Relaxed solution 1
    for token1, prob1 in D1.items():
        w2v_1 = get_embedding_matrix(token1)
        distances_word_to_sentences = np.zeros(L2)
        count = 0
        for token2, prob2 in D2.items():
            w2v_2 = get_embedding_matrix(token2)
            distances_word_to_sentences[count] = calc_euclidean(w2v_1, w2v_2)
            count += 1
        min_cost = np.min(distances_word_to_sentences)
        LR1 += min_cost * prob1

    # # Calculate the Relaxed solution 2
    for token2, prob2 in D2.items():
        w2v_2 = get_embedding_matrix(token2)
        distances_word_to_sentences = np.zeros(L1)
        count = 0
        for token1, prob1 in D1.items():
            w2v_1 = get_embedding_matrix(token1)
            distances_word_to_sentences[count] = calc_euclidean(w2v_1, w2v_2)
            count += 1
        min_cost = np.min(distances_word_to_sentences)
        LR2 += min_cost * prob2

    return np.max([LR1, LR2])


def Prefetch_and_prune(refDoc, docList, k):
    # refDoc is the referent document
    # docList is the list of documents to compare
    # k for the k-nn value

    n = len(docList)
    knn = np.zeros((k, 3))
    DIC = np.zeros((n, 3))

    for i in range(0, n):
        DIC[i][2] = i
        DIC[i][0] += WCD(refDoc, docList[i])

    # sort list of documents by WCD to referent document
    indSort = np.argsort(DIC, axis=0)
    DIC = DIC[indSort[:, 0], :]

    # calculate wmd for the knn
    for i in range(0, k):
        knn[i][2] += DIC[i][2]
        knn[i][1] += DIC[i][0]
        idx = DIC[i][2]
        idx = int(idx)
        knn[i][0] += WMD(refDoc, (docList[idx]))

    # Checking lower bound
    for i in range(k, n):

        currDoc = docList[int(DIC[i][2])]

        knn_current =  knn[k - 1][1]

        # 如果RWMD超过最近文档的WCD 就要剪枝
        diff = RWMD(refDoc, currDoc) -  knn[k - 1][1]

        if diff < 0:
            # skip to next iteration
            continue
        # prune up
        wmd_curr = WMD(refDoc, currDoc)
        DIC[i][1] = wmd_curr
        knn_to_add = [DIC[i][1], DIC[i][0], DIC[i][2]]
        knn = np.append(knn, [knn_to_add], axis=0)

        indSort_knn = np.argsort(knn, axis=0)
        knn = knn[indSort_knn[:, 0], :]
        knn = np.delete(knn, -1, 0)

    K = {}
    for i in range(k):
        idx = knn[i][2]
        idx = int(idx)
        # K[i] = " ".join(str(x) for x in docList[idx])
        K[idx] = docList[idx]

    return K


# print (WMD("people like this car", "those guys enjoy driving that"))
# print (WMD("hello", "hi"))
# print (WMD("Obama speaks to the media in Illinois", "The President greets the press in Chicago"))
# print (WMD("My father goes to work by bike", "My father rides to work"))
# print (WMD("Can you tell me the way to the zoo", "How can I get to the zoo"))

print (WMD("Obama speaks to the media in Illinois", "The President greets the press in Chicago"))
print (WCD("Obama speaks to the media in Illinois", "The President greets the press in Chicago"))
print (RWMD("Obama speaks to the media in Illinois", "The President greets the press in Chicago"))

## Prefetch_and_prune
sref='Obama speaks to the media in Illinois'
s1='The President greets the press in Chicago'
s2='The color of the big car is blue'
s3='Bush talks to journalists in Washington'
s4='The band gave a concert in Japan'
S_doc=list([s1,s2,s3,s4])

# S_doc=list([sref,s1,s2,s3,s4])
# S_doc = [removestop_and_split(value) for value in S_doc]
# sref = S_doc.pop(0)
# print(S_doc)
# print(sref)

start = timeit.default_timer()
print('Prefetch_and_prune',Prefetch_and_prune(sref, S_doc, 2))
stop = timeit.default_timer()
print('distance Prefetch_and_prune Time: ', stop - start)