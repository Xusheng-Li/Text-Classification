#!/usr/bin/env python
# coding: utf-8

# # 任务介绍
# 
# **任务：本次实践使用朴素贝叶斯方法解决文本分类问题**
# 
# **实践平台：百度AI实训平台-AI Studio、python3.7**
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/3f746f16e75242c5b176e93c93b3122ed67a13057b2f4b99811b03c95a745723)
# 

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# # 数据集介绍
# 网上公开中文新闻数据：
# 
# 数据来源：从中文新闻网站上爬取56821条新闻摘要数据。
# 
# 数据内容：数据集中包含10个类别
# 
# 数据划分：本次实践将其中90%作为训练集，10%作为验证集。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/f1b298fb982b4cec9a3ee66b8d0b67d497995f140de142ec9a459914eace6931)
# 
# 

# # 模型选择
# **贝叶斯分类：** 贝叶斯分类是一类分类算法的总称，这类算法均以贝叶斯定理为基础，故统称为贝叶斯分类。
# 
# **贝叶斯公式:**    $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
# 
# 在文本分类（classification）问题中，我们要将一个句子分到某个类别，我们将句子中的词或字视为句子的属性，因此一个句子是由个属性（字/词）组成的，把众多属性看做一个向量，即X=(x1,x2,x3,…,xn)，用X这个向量来代表这个句子。
# 类别也有很多种，我们用集合Y={y1,y2,…ym}表示。
# 
# 一个句子属于yk类别的概率可以表示为：$P(Y=yk|X=(x1,x2,x3,…,xn))$
# 
# 如果一个句子的属性向量X属于yk类别的概率最大:
# 
# 即,  $P(Y=yk|X) = max{P(Y=y1|X),P(Y=y2|X),P(Y=y3|X),...,P(Y=ym|X)}$,
# 
# 其中，X=(x1,x2,x3,…,xn)
# 
# 就可以给X打上yk标签，意思是说X属于yk类别。这就是所谓的分类(Classification)。
# 
# **朴素贝叶斯：**
# 
# 假设X=(x1,x2,x3,…,xn)中的所有属性都是独立的，
# 
# 即$P(Y=yk|X=(x1,x2,x3,…,xn)) = \frac{P(x1｜Y=yk)P(x2｜Y=yk)...P(xn｜Y=yk)P(Y=yk)}{P(x1)P(x2)...P(xn)}$
# 
# **拉普拉斯平滑的引入：**
# 
# 如果某个属性的条件概率为0，则会导致整体概率为零，为了避免这种情况出现，引入拉普拉斯平滑参数，即将条件概率为0的属性的概率设定为固定值。

# # 整体流程
# 1、数据准备：
# * 数据预处理：对去除文本中的标点符号，并对句子进行分词。
# * 生成词典：一个词即为一个特征，统计所有句子中出现过的词，去除停用词，并统计词频，形成一个词典。
# * 确定特征词：从词典中选择一部分词作为特征词。
# * 形成特征向量：利用特征词，将每一个句子转化为特征向量。
# 
# 2、训练分类器模型
# 
# 3、评估训练效果
# 
# 4、使用模型进行预测

# **关于特征词和特征向量：**
# 
# **【举例说明】**
# 
# 停用词：是、的、你、我、他，这、那
# 
# 特征词：[‘中国’，’西安‘，’天安门‘，’首都‘，’故宫’，‘机器学习’，’北京‘]
# 
# text：北京是中国的首都
# 
# 通过分词之后 [北京，是，中国，的，首都]
# 
# 去除停用词：[北京，中国，首都]
# 
# 形成特征向量：[1,0,0,1,0,0,1]
# 

# In[2]:


#导入必要的包
import random
import jieba  # 处理中文
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
import re,string


# In[3]:


def text_to_words(file_path):
    '''
    分词
    return:sentences_arr, lab_arr
    '''
    sentences_arr = []
    lab_arr = []
    with open(file_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            lab_arr.append(line.split('_!_')[1])
            sentence = line.split('_!_')[-1].strip()
            sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）《》：]+", "",sentence) #去除标点符号
            sentence = jieba.lcut(sentence, cut_all=False)
            sentences_arr.append(sentence)
    return sentences_arr, lab_arr


# In[4]:



def load_stopwords(file_path):
    '''
    创建停用词表
    参数 file_path:停用词文本路径
    return：停用词list
    '''
    stopwords = [line.strip() for line in open(file_path, encoding='UTF-8').readlines()]
    return stopwords



# In[5]:



def get_dict(sentences_arr,stopswords):
    '''
    遍历数据，去除停用词，统计词频
    return: 生成词典
    '''
    word_dic = {}
    for sentence in sentences_arr:
        for word in sentence:
            if word != ' ' and word.isalpha():
                if word not in stopswords:
                    word_dic[word] = word_dic.get(word,1) + 1
    word_dic=sorted(word_dic.items(),key=lambda x:x[1],reverse=True) #按词频序排列

    return word_dic


# In[6]:



def get_feature_words(word_dic,word_num):
    '''
    从词典中选取N个特征词，形成特征词列表
    return: 特征词列表
    '''
    n = 0
    feature_words = []
    for word in word_dic:
        if n < word_num:
            feature_words.append(word[0])
        n += 1
    return feature_words


# In[7]:


# 文本特征
def get_text_features(train_data_list, test_data_list, feature_words):
    '''
    根据特征词，将数据集中的句子转化为特征向量
    '''
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words] # 形成特征向量
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


# In[8]:


#获取分词后的数据及标签
sentences_arr, lab_arr = text_to_words('data/data6826/news_classify_data.txt')
#加载停用词
stopwords = load_stopwords('data/data43470/stopwords_cn.txt')
# 生成词典
word_dic = get_dict(sentences_arr,stopwords)
#数据集划分
train_data_list, test_data_list, train_class_list, test_class_list = model_selection.train_test_split(sentences_arr, 
                                                                                                      lab_arr, 
                                                                                                      test_size=0.1)
#生成特征词列表
feature_words =  get_feature_words(word_dic,10000)

#生成特征向量
train_feature_list,test_feature_list = get_text_features(train_data_list,test_data_list,feature_words)


# In[15]:


from sklearn.metrics import accuracy_score,classification_report
#获取朴素贝叶斯分类器
classifier = MultinomialNB(alpha=1.0,  # 拉普拉斯平滑
                          fit_prior=True,  #否要考虑先验概率
                          class_prior=None)

#进行训练                        
classifier.fit(train_feature_list, train_class_list)
# 在验证集上进行验证
predict = classifier.predict(test_feature_list)
test_accuracy = accuracy_score(predict,test_class_list)
print("accuracy_score: %.4lf"%(test_accuracy))
print("Classification report for classifier:\n",classification_report(test_class_list, predict))


# In[11]:


#加载句子，对句子进行预处理
def load_sentence(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）《》：]+", "",sentence) #去除标点符号
    sentence = jieba.lcut(sentence, cut_all=False)
    return sentence


# In[12]:


lab = [ '文化', '娱乐', '体育', '财经','房产', '汽车', '教育', '科技', '国际', '证券']

p_data = '【中国稳健前行】应对风险挑战必须发挥制度优势'
sentence = load_sentence(p_data)
sentence= [sentence]
print('分词结果:', sentence)
#形成特征向量
p_words = get_text_features(sentence,sentence,feature_words)
res = classifier.predict(p_words[0])
print("所属类型：",lab[int(res)])

