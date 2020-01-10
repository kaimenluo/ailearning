#读取数据初始化字典
word2id,id2word = {},{}
tag2id,id2tag = {},{}

for line in open("traindata.txt"):
    item = line.split("/")
    word, tag = item[0], item[1].rstrip()
    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word

    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(id2tag)] = tag

#初始化pi 转移矩阵 发射矩阵
import numpy as np
N = len(tag2id)
M = len(word2id)

pi = np.zeros(N)
A = np.zeros((N,N)) #观测矩阵
B = np.zeros((N,M)) #状态转移矩阵

# 统计参数
prev_tag_id = -1
for line in open("traindata.txt"):
    item = line.split("/")
    word, tag = item[0], item[1].rstrip()
    word_id,tag_id = word2id[word],tag2id[tag]

    if prev_tag_id == -1:#句首
        pi[tag_id] += 1
        B[tag_id][word_id] += 1
    else:
        B[tag_id][word_id] += 1
        A[prev_tag_id][tag_id] += 1

    if word == ".": #表示句子结束了
        prev_tag_id = -1
    else:
        prev_tag_id = tag_id


# 转换成概率
pi = pi / sum(pi)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])

#log 函数
def log(v):
    if v == 0:
        return np.log(v + 0.000001)
    return np.log(v)

def viterbi(observe, pi, A, B):
    x = [word2id[value] for value in observe.split(" ") if value in word2id]
    T = len(x)
    dp = np.zeros((T, N))
    ptr = np.zeros((T, N), dtype=int)

    #初始化状态
    for j in range(N):
        dp[0][j] = log(pi[j]) + log(B[j][x[0]])

    for i in range(1,T):
        for j in range(N):
            prob, k = max([(dp[i-1][k] + log(A[k][j]) + log(B[j][x[i]]), k) for k in range(N)])
            dp[i][j] = prob
            ptr[i][j] = k


    # best_seq = np.zeros(T)
    best_seq = [0] * T
    best_seq[T-1] = np.argmax(dp[T-1])

    # 根据下一步推断上一步的状态
    for i in range(T-2, -1, -1):
        best_seq[i] = ptr[i+1][best_seq[i+1]]

    for i in range(T):
        print(id2tag[best_seq[i]])

if __name__ == "__main__":
    x = "keep new to everything"
    viterbi(x, pi, A, B)

    print("\n")

    x = "it is amazing"
    viterbi(x, pi, A, B)