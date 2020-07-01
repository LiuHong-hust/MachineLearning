import os
import shutil                           # 移动文件
import random                           # 随机化抽取文件
import matplotlib.pyplot as plt         # 画图
from nltk.corpus import stopwords       # 去停用词

cachedStopWords = stopwords.words("english")    # 英文停用词

def fileWalker(path):
    # 遍历语料目录，将所有语料文件绝对路径存入列表fileArray
    fileArray = []      #文件列表
    for root, dirs, files in os.walk(path):
        for fn in files:
            eachpath = str(root+'\\'+fn)
            fileArray.append(eachpath)
    return fileArray


def test_set_select():
    # 从spam和ham集中随机选10封移动到test集中作为测试集
    filepath = r"D:\2020chun\jimail\english_email"
    testpath = r"D:\2020chun\jimail\english_email\test"
    files = fileWalker(filepath)
    random.shuffle(files)
    top10 = files[:10]
    for ech in top10:
        ech_name = testpath+'\\'+('_'.join(ech.split('\\')[-2:]))
        shutil.move(ech, testpath)
        os.rename(testpath+'\\'+ech.split('\\')[-1], ech_name)
    return


def test_set_clear():
    # 移动test测试集中文件回spam和ham中，等待重新抽取测试集
    filepath = r"D:\2020chun\jimail\english_email"
    testpath = r"D:\2020chun\jimail\english_email\test"
    files = fileWalker(testpath)
    for ech in files:
        ech_initial = filepath + '\\' + '\\'.join(' '.join(ech.split('\\')[-1:]).split('_'))
        ech_move = filepath + '\\' + (' '.join(ech.split('\\')[-1:]).split('_'))[0]
        shutil.move(ech, ech_move)
        os.rename(ech_move+'\\'+' '.join(ech.split('\\')[-1:]), ech_initial)
    return


def readtxt(path, encoding):
    # 按encoding方式按行读取path路径文件所有行，返回行列表lines
    with open(path, 'r',encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
    return lines


def email_parser(email_path):
    # 去特殊字符标点符号，返回纯单词列表clean_word
    punctuations = """,.<>()*&^%$#@!'";~`[]{}|、\\/~+_-=?"""
    content_list = readtxt(email_path, 'gbk')
    content = (' '.join(content_list)).replace('\r\n', ' ').replace('\t', ' ')
    clean_word = []
    for punctuation in punctuations:
        content = (' '.join(content.split(punctuation))).replace('  ', ' ')
        clean_word = [word.lower()
                      for word in content.split(' ') if word.lower() not in cachedStopWords and len(word) > 2]
        # 去除停用词
    return clean_word


def get_word(email_file):
    # 获取email_file路径下所有文件的总单词列表，append入word_list，extend入word_set并去重转为set
    word_list = []
    word_set = []
    email_paths = fileWalker(email_file)
    for email_path in email_paths:
        clean_word = email_parser(email_path)
        word_list.append(clean_word)
        word_set.extend(clean_word)
    return word_list, set(word_set)


def count_word_prob(email_list, union_set):
    # 返回训练集词频字典word_prob
    word_prob = {}
    for word in union_set:
        counter = 0
        for email in email_list:
            if word in email:
                counter += 1
            else:
                continue
        prob = 0.0
        if counter != 0:
            prob = counter/len(email_list)
        else:
            prob = 0.05  # 进在某一分类中未出现则令该分类下该词词频TF=0.01，0.05，……，越大越会把spam误判成ham
        word_prob[word] = prob
    return word_prob


def filter(ham_word_pro, spam_word_pro, test_file):
    # 进行一次对测试集(10封邮件)的测试，输出对测试集的判断结果
    # 并返回准确率right_rate，以及把spam误判成ham和总误判次数对应情况
    right = 0
    wrong = 0
    wrong_spam = 0
    test_paths = fileWalker(test_file)
    for test_path in test_paths:
        # 贝叶斯推断计算与判别实现
        email_spam_prob = 0.0
        spam_prob = 0.5  # 假设P(spam) = 0.5
        ham_prob = 0.5  # P(ham) = 0.5
        file_name = test_path.split('\\')[-1]
        prob_dict = {}
        words = set(email_parser(test_path))
        for word in words:  # 统计测试集所出现单词word的P(spam|word)
            Psw = 0.0
            if word not in spam_word_pro:
                Psw = 0.4  # 第一次出现的新单词设P(spam|new word) = 0.4 by Paul Graham
            else:
                Pws = spam_word_pro[word]  # P(word|spam)
                Pwh = ham_word_pro[word]  # P(word|ham)
                Psw = spam_prob*(Pws/(Pwh*ham_prob+Pws*spam_prob))
                # P(spam|word) = P(spam)*P(word|spam)/P(word)
                #              = P(spam)*P(word|spam)/(P(word|ham)*P(ham)+P(word|spam)*P(spam))
            prob_dict[word] = Psw
        numerator = 1
        denominator_h = 1
        for k, v in prob_dict.items():
            numerator *= v          # P1P2…Pn = P(spam|word1)*P(spam|word2)*…*P(spam|wordn)
            denominator_h *= (1-v)  # (1-P1)(1-P2)…(1-Pn) = (1-P(spam|word1))*(1-P(spam|word2))*…*(1-P(spam|wordn))
        email_spam_prob = round(numerator/(numerator+denominator_h), 4)
        # P(spam|word1word2…wordn) = P1P2…Pn/(P1P2…Pn+(1-P1)(1-P2)…(1-Pn))
        if email_spam_prob > 0.9:  # P(spam|word1word2…wordn) > 0.9 认为是spam垃圾邮件
            print(file_name, '   spam      ', email_spam_prob)
            if file_name.split('_')[1] == '25.txt':
                print(prob_dict)
            if file_name.split('_')[0] == 'spam':  # 记录是否判断准确
                right += 1
            else:
                wrong += 1
                print('Wrong Prediction')
        else:
            print(file_name, '   ham      ', email_spam_prob)
            if file_name.split('_')[1] == '25.txt':
                print(prob_dict)
            if file_name.split('_')[0] == 'ham':  # 记录是否判断准确
                right += 1
            else:
                wrong += 1
                wrong_spam += 1  # 记录把spam误判成ham的次数
                print('Wrong Prediction')

        # print(prob_dict)
    right_rate = right/(right+wrong)  # 计算一个测试集的准确率
    if wrong != 0:
        wrong_spam_rate = [wrong_spam, wrong]  # [把spam误判成ham的次数，总误判次数]
    else:
        wrong_spam_rate = [-1]  # 表示总误判次数为0
    return right_rate, wrong_spam_rate


def main():
    # 主函数
    right_rate_list = []
    wrong_spam_rate_list = []
    ham_file = r"D:\2020chun\jimail\english_email\ham"
    spam_file = r"D:\2020chun\jimail\english_email\spam"
    test_file = r"D:\2020chun\jimail\english_email\test"
    for i in range(100):
        # 进行100次抽取测试集，测试并记录准确率，注意训练集应不包含测试集
        print('\n第',i+1,'次:')
        test_set_select()  # 构造测试集
        ham_list, ham_set = get_word(ham_file)
        spam_list, spam_set = get_word(spam_file)
        union_set = ham_set | spam_set  # 合并纯单词集合
        ham_word_pro = count_word_prob(ham_list, union_set)  # 单词在ham中的出现频率字典
        spam_word_pro = count_word_prob(spam_list, union_set)  # 单词在spam里的出现频率字典
        print('文件名        分类结果      垃圾邮件概率')
        rig, wrg = filter(ham_word_pro, spam_word_pro, test_file)
        right_rate_list.append(rig)  # 返回正确率
        wrong_spam_rate_list.append(wrg)  # 返回误报spam->ham占比
        test_set_clear()  # 还原测试集
    # 画出100次判别的准确率散点图
    x = range(100)
    y = right_rate_list
    plt.scatter(x, y)
    plt.title('Correct Rate of 100 Times')
    plt.show()
    # 输出100次误报spam->ham占比列表
    print(wrong_spam_rate_list)
    return


if __name__ == '__main__':
    main()

input()
