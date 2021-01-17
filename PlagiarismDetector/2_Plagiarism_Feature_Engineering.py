#!/usr/bin/env python
# coding: utf-8

# # 剽窃检测，特征工程
# 
# 在此项目中，你需要构建一个剽窃检测器，它会检测答案文本文件并进行二元分类：根据文本文件与提供的原文之间的相似度，将文件标为剽窃文件或非剽窃文件。
# 
# 你的第一个任务是创建一些可以用于训练分类模型的特征。这个任务将分为以下几个步骤：
# 
# * 清理和预处理数据。
# * 定义用于比较答案和原文之间相似性的特征，并提取相似性特征。
# * 通过分析不同特征之间的相关性，选择“合适的”特征。
# * 创建训练/测试 `.csv` 文件，其中包含训练/测试数据点的相关特征和类别标签。
# 
# 在下个 notebook (Notebook 3) 中，你将使用在此 notebook 中创建的这些特征和 `.csv` 文件在 SageMaker notebook 实例中训练一个二元分类模型。
# 
# 你将根据[这篇论文](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c412841_developing-a-corpus-of-plagiarised-short-answers/developing-a-corpus-of-plagiarised-short-answers.pdf)中的说明定义几个不同的相似性特征，这些特征将帮助你构建强大的剽窃检测器。
# 
# 在完成此 notebook 时，你需要完成此 notebook 中的所有练习并回答所有问题。
# > 所有任务都标有**练习**，所有问题都标有**问题**。
# 
# 你需要决定在最终训练和测试数据中包含哪些特征。
# 
# ---

# ## 读取数据
# 
# 以下单元格将下载必要的项目数据并将文件解压缩到文件夹 `data/` 中。
# 
# 此数据是谢菲尔德大学 Paul Clough（信息研究）和 Mark Stevenson（计算机科学）创建的数据集的稍加修改版本。要了解数据收集和语料库信息，请访问[谢菲尔德大学网站](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html). 
# 
# > **数据引用**：Clough, P. and Stevenson, M. Developing A Corpus of Plagiarised Short Answers, Language Resources and Evaluation: Special Issue on Plagiarism and Authorship Analysis, In Press. [下载]

# In[1]:


# NOTE:
# you only need to run this cell if you have not yet downloaded the data
# otherwise you may skip this cell or comment it out

get_ipython().system('wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip')
get_ipython().system('unzip data')


# In[4]:


# import libraries
import pandas as pd
import numpy as np
import os


# 这个剽窃数据集由多个文本文件组成；每个文件的特性都在名为 `file_information.csv` 的 `.csv` 文件中进行了总结，我们可以使用 `pandas` 读取该文件。

# In[3]:


csv_file = 'data/file_information.csv'
plagiarism_df = pd.read_csv(csv_file)

# print out the first few rows of data info
plagiarism_df.head()


# ## 剽窃类型
# 
# 每个文本文件都有一个相关**任务**（任务 A-E）和一个剽窃**类别**，从上述 DataFrame 中可以看出。
# 
# ###  A-E 五种任务
# 
# 每个文本文件都包含一个简短问题的答案；这些问题标为任务 A-E。例如任务 A 的问题是：“面向对象编程中的继承是什么意思？”
# 
# ### 剽窃类别
# 
# 每个文本文件都有相关的剽窃标签/类别：
# 
# **1. 剽窃类别：`cut`、`light` 和 `heavy`。**
# * 这些类别表示不同级别的剽窃答案文本。`cut` 类答案直接复制了原文，`light` 类答案在原文的基础上稍微改写了一下，而 `heavy` 类答案以原文为基础，但是改写程度很大（可能是最有挑战性的剽窃检测类型）。
#      
# **2. 非剽窃类别：`non`。** 
# * `non` 表示答案没有剽窃，没有参考维基百科原文。
#     
# **3. 特殊的原文类别：`orig`。**
# * 这是原始维基百科文本对应的一种类别。这些文件仅作为比较基准。

# ---
# ## 预处理数据
# 
# 在下面的几个单元格中，你将创建一个新的 DataFrame，其中包含关于 `data/` 目录下所有文件的相关信息。该 DataFrame 可以为特征提取和训练二元剽窃分类器准备好数据。

# ### 练习：将类别转换为数值数据
# 
# 你将发现数据集中的 `Category` 列包含字符串或类别值，要为特征提取准备这些特征，我们需要将类别转换为数值。此外，我们的目标是创建一个二元分类器，所以我们需要一个二元类别标签，可以表示答案文本是剽窃文件 (1) 还是非剽窃文件 (0)。请完成以下函数 `numerical_dataframe`，它会根据名称读取 `file_information.csv`，并返回新的 DataFrame，其中包含一个数值 `Category` 列，以及新的 `Class` 列，该列会将每个答案标为剽窃文件或非剽窃文件。
# 
# 你的函数应该返回一个具有以下属性的新 DataFrame：
# 
# * 4 列：`File`、`Task`、`Category`、`Class`。`File` 和 `Task` 列可以与原始 `.csv` 文件一样。
# * 根据以下规则将所有 `Category` 标签转换为数值标签（更高的值表示更高级别的剽窃行为）：
#     * 0 = `non`
#     * 1 = `heavy`
#     * 2 = `light`
#     * 3 = `cut`
#     * -1 = `orig`，这是表示原始文件的特殊值。
# * 对于新的 `Class` 列
#     * 任何非剽窃 (`non`) 答案文本的类别标签都应为 `0`。 
#     * 任何剽窃类答案文本的类别标签都应为 `1`。
#     * 任何 `orig` 文本都对应特殊标签 `-1`。 
# 
# ### 预期输出
# 
# 运行函数后，应该获得行如下所示的 DataFrame：
# ```
# 
#         File	     Task  Category  Class
# 0	g0pA_taska.txt	a	  0   	0
# 1	g0pA_taskb.txt	b	  3   	1
# 2	g0pA_taskc.txt	c	  2   	1
# 3	g0pA_taskd.txt	d	  1   	1
# 4	g0pA_taske.txt	e	  0	   0
# ...
# ...
# 99   orig_taske.txt    e     -1      -1
# 
# ```

# In[5]:


# Read in a csv file and return a transformed dataframe
def numerical_dataframe(csv_file='data/file_information.csv'):
    '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
       This function does two things: 
       1) converts `Category` column values to numerical values 
       2) Adds a new, numerical `Class` label column.
       The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
       Source texts have a special label, -1.
       :param csv_file: The directory for the file_information.csv file
       :return: A dataframe with numerical categories and a new `Class` label column'''
    
    # your code here
    df = pd.read_csv(csv_file)
    df.loc[:,'Class'] =  df.loc[:,'Category'].map({'non': 0, 'heavy': 1, 'light': 1, 'cut': 1, 'orig': -1})
    df.loc[:,'Category'] =  df.loc[:,'Category'].map({'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1})
    
    return df


# ### 测试单元格
# 
# 下面是几个测试单元格。第一个是非正式测试，你可以通过调用你的函数并输出返回的结果，检查代码是否符合预期。
# 
# 下面的**第二个**单元格是更严格的测试单元格。这样的单元格旨在确保你的代码能按预期运行，并形成可能会在后面的测试/代码中使用的任何变量，在这里是指 dataframe `transformed_df`。
# 
# > 你应该按前后顺序（出现在 notebook 中的顺序）运行此 notebook 中的单元格。对于测试单元格来说，这一点很重要。
# 
# 通常，后面的单元格依赖于在之前的单元格中定义的函数、导入项或变量。例如，某些测试需要依赖于前面的测试才能运行。
# 
# 这些测试并不能测试所有情况，但是可以很好地检查你的代码是否正确。

# In[6]:


# informal testing, print out the results of a called function
# create new `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

# check work
# check that all categories of plagiarism have a class label = 1
transformed_df.head(10)


# In[7]:


# test cell that creates `transformed_df`, if tests are passed

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

# importing tests
import problem_unittests as tests

# test numerical_dataframe function
tests.test_numerical_df(numerical_dataframe)

# if above test is passed, create NEW `transformed_df`
transformed_df = numerical_dataframe(csv_file ='data/file_information.csv')

# check work
print('\nExample data: ')
transformed_df.head()


# ## 文本处理和拆分数据
# 
# 这个项目的目标是构建一个剽窃分类器。本质上这个任务是比较文本；查看给定答案和原文，比较二者并判断答案是否剽窃了原文。要有效地进行比较并训练分类器，我们需要完成几项操作：预处理所有文本数据并准备文本文件（在此项目中有 95 个答案文件和 5 个原始文件），使文件更容易比较，并将数据划分为 `train` 和 `test` 集合，从而分别可以用于训练分类器和评估分类器。
# 
# 为此，我们向你提供了可以向上面的 `transformed_df` 添加额外信息的代码。下面的两个单元格不需要更改；它们会向 `transformed_df` 添加两列：
# 
# 1. 一个 `Text` 列；此列包含 `File` 的所有小写文本，并删除了多余的标点。
# 2. 一个 `Datatype` 列；它是一个字符串值 `train`、`test` 或 `orig`，将数据点标记为训练集或测试集。
# 
# 你可以在项目目录的 `helpers.py` 文件中找到如何创建这些额外列的详细信息。建议通读该文件，了解文本是如何处理的，以及数据是如何拆分的。
# 
# 请运行以下单元格以获得 `complete_df`，其中包含剽窃检测和特征工程所需的所有信息。

# In[8]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import helpers 

# create a text column 
text_df = helpers.create_text_column(transformed_df)
text_df.head()


# In[9]:


# after running the cell above
# check out the processed text for a single file, by row index
row_idx = 0 # feel free to change this index

sample_text = text_df.iloc[0]['Text']

print('Sample processed text:\n\n', sample_text)


# ## 将数据拆分成训练集和测试集
# 
# 下个单元格将向给定的 DataFrame 添加一个 `Datatype` 列，表示记录是否是：
# * `train` - 训练数据，用于训练模型。
# * `test` - 测试数据，用于评估模型。
# * `orig` - 任务的原始维基百科文件。
# 
# ### 分层抽样
# 
# 给定的代码使用了辅助函数，你可以在主项目目录的 `helpers.py` 文件中查看该函数。该函数实现了[分层随机抽样](https://en.wikipedia.org/wiki/Stratified_sampling)，可以按照任务和剽窃量随机拆分数据。分层抽样可以确保获得在任务和剽窃组合之间均匀分布的训练和测试数据。约 26% 的数据作为测试集，约 74% 的数据作为训练集。
# 
# 函数 **train_test_dataframe** 接受一个 DataFrame，并假设该 DataFrame 具有 `Task` 和 `Category` 列，然后返回一个修改过的 DataFrame，表示文件属于哪种 `Datatype`（训练、测试或原始文件）。抽样方式将根据传入的 *random_seed* 稍微不同。犹豫样本量比较小，所以这种分层随机抽样可以为二元剽窃分类器提供更稳定的结果。稳定性是指在给定随机 seed 后，分类器的准确率方差更小。

# In[10]:


random_seed = 1 # can change; set for reproducibility

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import helpers

# create new df with Datatype (train, test, orig) column
# pass in `text_df` from above to create a complete dataframe, with all the information you need
complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

# check results
complete_df.head(10)


# # 判断剽窃行为
# 
# 准备好数据并创建了包含信息的 `complete_df`（包括与每个文件相关的文本和类别）后，可以继续完成下个任务了，即提取可以用于剽窃分类的相似性特征。
# 
# > 注意：以下代码练习不会修改假设现在存在的 `complete_df` 的现有列。
# 
# `complete_df` 应该始终包含以下列：`['File', 'Task', 'Category', 'Class', 'Text', 'Datatype']`。你可以添加其他列，并且可以通过复制 `complete_df` 的部分内容创建任何新的 DataFrames，只要不直接修改现有值即可。
# 
# ---

# 
# # 相似性特征
# 
# 剽窃检测的一种方式是计算**相似性特征**，这些特征可以衡量给定文本与原始维基百科原文之间的相似性（对于特定的任务 A-E 来说）。你可以根据[这篇剽窃检测论文](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c412841_developing-a-corpus-of-plagiarised-short-answers/developing-a-corpus-of-plagiarised-short-answers.pdf)创建相似性特征。
# > 在这篇论文中，研究人员创建了叫做**包含系数**和**最长公共子序列**的特征。
# 
# 你将使用这些特征作为输入，训练模型区分剽窃文本和非剽窃文本。
# 
# ## 特征工程
# 
# 下面深入讨论下我们要包含在剽窃检测模型中的特征，以及如何计算这些特征。在下面的解释部分，我将提交的文本文件称为**学员答案文本 (A)**，并将原始维基百科文件（我们要将答案与之比较的文件）称为**维基百科原文 (S)**。
# 
# ### 包含系数
# 
# 你的第一个任务是创建**包含系数特征**。为了理解包含系数，我们先回顾下 [n-gram](https://en.wikipedia.org/wiki/N-gram) 的定义。*n-gram* 是一个序列字词组合。例如，在句子“bayes rule gives us a way to combine prior knowledge with new information”中，1-gram 是一个字词，例如“bayes”。2-gram 可以是“bayes rule”，3-gram 可以是“combine prior knowledge”。
# 
# > 包含系数等于维基百科原文 (S) 的 n-gram 字词计数与学员答案文本 (S) 的 n-gram 字词计数之间的**交集**除以学员答案文本的 n-gram 字词计数。
# 
# $$ \frac{\sum{count(\text{ngram}_{A}) \cap count(\text{ngram}_{S})}}{\sum{count(\text{ngram}_{A})}} $$
# 
# 如果两段文本没有公共的 n-gram，那么包含系数为 0，如果所有 n-gram 都有交集，那么包含系数为 1。如果有更长的 n-gram 是一样的，那么可能存在复制粘贴剽窃行为。在此项目中，你需要决定在最终项目中使用什么样的 `n` 或多个 `n`。
# 
# ### 练习：创建包含系数特征
# 
# 根据你创建的 `complete_df`，你应该获得了比较学员答案文本 (A) 与对应的维基百科原文 (S) 所需的所有信息。任务 A 的答案应该与任务 A 的原文进行比较，并且任务 B、C、D 和 E 的答案应该与对应的原文进行比较。
# 
# 在这道练习中，你需要完成函数 `calculate_containment`，它会根据以下参数计算包含系数：
# * 给定 DataFrame `df`（假设为上面的 `complete_df`）
# * `answer_filename`，例如 'g0pB_taskd.txt' 
# * n-gram 长度 `n`
# 
# ### 计算包含系数
# 
# 完成此函数的一般步骤如下所示：
# 1. 根据给定 `df` 中的所有文本文件创建一个 n-gram 计数数组；建议使用 [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)。
# 2. 获得给定 `answer_filename` 的已处理答案和原文。
# 3. 根据以下公式计算答案和原文之间的包含系数。
# 
#     >$$ \frac{\sum{count(\text{ngram}_{A}) \cap count(\text{ngram}_{S})}}{\sum{count(\text{ngram}_{A})}} $$
#     
# 4. 返回包含系数值。
# 
# 在完成以下函数时，可以编写任何辅助函数。

# In[13]:


from sklearn.feature_extraction.text import CountVectorizer

# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(df, n, answer_filename):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text, 
       and its associated source text, and calculating the normalized intersection of those counts.
       :param df: A dataframe with columns,
           'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
       :param n: An integer that defines the ngram size
       :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
       :return: A single containment value that represents the similarity
           between an answer text and its source text.
    '''
    
    answer_text, answer_task  = df[df.File == answer_filename][['Text', 'Task']].iloc[0]  
    source_text = df[(df.Task == answer_task) & (df.Class == -1)]['Text'].iloc[0]
    
    counts = CountVectorizer(analyzer='word', ngram_range=(n,n))
    ngrams_array = counts.fit_transform([answer_text, source_text]).toarray()
    containment = (np.minimum(ngrams_array[0],ngrams_array[1]).sum())/(ngrams_array[0].sum())
    
    return containment


# ### 测试单元格
# 
# 实现了包含系数函数后，你可以测试函数行为。
# 
# 以下单元格将遍历前几个文件，并根据指定的 n 和文件计算原始类别和包含系数值。
# 
# >如果你正确实现了该函数，你应该看到非剽窃类别的包含系数值很低或接近 0，剽窃示例的包含系数更高，或接近 1。
# 
# 注意当 n 的值改变时会发生什么。建议将代码应用到多个文件上，并比较生成的包含系数。你应该看到，最高的包含系数对应于最高剽窃级别 (`cut`) 的文件。

# In[14]:


# select a value for n
n = 3

# indices for first few files
test_indices = range(5)

# iterate through files and calculate containment
category_vals = []
containment_vals = []
for i in test_indices:
    # get level of plagiarism for a given file index
    category_vals.append(complete_df.loc[i, 'Category'])
    # calculate containment for given file and n
    filename = complete_df.loc[i, 'File']
    c = calculate_containment(complete_df, n, filename)
    containment_vals.append(c)

# print out result, does it make sense?
print('Original category values: \n', category_vals)
print()
print(str(n)+'-gram containment values: \n', containment_vals)


# In[15]:


# run this test cell
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# test containment calculation
# params: complete_df from before, and containment function
tests.test_containment(complete_df, calculate_containment)


# ### 问题 1：为何我们可以在为创建模型而拆分 DataFrame 之前，计算所有数据（训练和测试数据）的包含系数特征？也就是说，为何在计算包含系数时，测试数据和训练数据不会相互影响？

# **回答：**
# 
# 数据预处理阶段还没有建立模型，并且包含系数是文本的一种独立特征，因此测试数据和训练数据不会相互影响。

# ---
# ## 最长公共子序列
# 
# 包含系数是检测两个文档用词重叠现象的很好方式；还有助于发现剪切-粘贴和改写级别的剽窃行为。因为剽窃是一项很复杂的任务，有多种不同的级别，所以通常有必要包含其他相似性特征。这篇论文还讨论了**最长公共子序列**特征。
# 
# > 最长公共子序列是指维基百科原文 (S) 和学员答案文本 (A) 之间一样的最长字词（或字母）字符串。该值也会标准化，即除以学员答案文本中的总字词（字母）数量。 
# 
# 在这道练习中，你的任务是计算两段文本之间的最长公共字词子序列。
# 
# ### 练习：计算最长公共子序列
# 
# 请完成函数 `lcs_norm_word`；它应该会计算学员答案文本与维基百科原文之间的最长公共子序列。
# 
# 举个具体的例子比较好理解。最长公共子序列 (LCS) 问题可能如下所示：
# * 假设有两段文本：长度为 n 的文本 A（答案）和长度为 m 的字符串 S（原文）。我们的目标是生成最长公共子序列：在两段文本中都出现过的从左到右最长公共子序列（字词不需要连续出现）。
# * 有两句话：
#     * A = "i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents"
#     * S = "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents"
# 
# * 在此示例中，我们发现每句话的开头比较相似，有重叠的一串字词“pagerank is a link analysis algorithm used by”，然后稍微有所变化。我们**继续从左到右地在两段文本中移动**，直到看到下个公共序列；在此例中只有一个字词“google”。接着是“that”和“a”，最后是相同的结尾“to each element of a hyperlinked set of documents”。
# * 下图演示了这些序列是如何在每段文本中按顺序发现的。
# 
# <img src='notebook_ims/common_subseq_words.png' width=40% />
# 
# * 这些字词按顺序在每个文档中从左到右地出现，虽然中间夹杂着一些字词，我们也将其看做两段文本的最长公共子序列。
# * 统计出公共字词的数量为 20。**所以，LCS 的长度为 20**。
# * 接下来标准化该值，即除以学员答案的总长度；在此例中，长度仅为 27。**所以，函数 `lcs_norm_word` 应该返回值 `20/27` 或约 `0.7408`**。
# 
# 所以，LCS 可以很好地检测剪切-粘贴剽窃行为，或检测某人是否在答案中多次参考了相同的原文。

# ### LCS，动态规划
# 
# 从上面的示例可以看出，这个算法需要查看两段文本并逐词比较。你可以通过多种方式解决这个问题。首先，可以使用 `.split()` 将每段文本拆分成用逗号分隔的字词列表，以进行比较。然后，遍历文本中的每个字词，并在比较过程中使 LCS 递增。
# 
# 在实现有效的 LCS 算法时，建议采用矩阵和动态规划法。**动态规划**是指将更大的问题拆分成一组更小的子问题，并创建一个完整的解决方案，不需要重复解决子问题。
# 
# 这种方法假设你可以将大的 LCS 任务拆分成更小的任务并组合起来。举个简单的字母比较例子：
# 
# * A = "ABCD"
# * S = "BD"
# 
# 一眼就能看出最长的字母子序列是 2（B 和 D 在两个字符串中都按顺序出现了）。我们可以通过查看两个字符串 A 和 S 中每个字母之间的关系算出这个结果。
# 
# 下图是一个矩阵，A 的字母位于顶部，S 的字母位于左侧：
# 
# <img src='notebook_ims/matrix_1.png' width=40% />
# 
# 这个矩阵的列数和行数为字符串 S 和 A 中的字母数量再加上一行和一列，并在顶部和左侧填充了 0。所以现在不是 2x4 矩阵，而是 3x5 矩阵。
# 
# 下面将问题拆分成更小的 LCS 问题并填充矩阵。例如，先查看最短的子字符串：A 和 S 的起始字母。“A”和“B”这两个字母之间最长的公共子序列是什么？
# 
# **答案是 0，在相应的单元格里填上 0。**
# 
# <img src='notebook_ims/matrix_2.png' width=30% />
# 
# 然后看看下个问题，“AB”和“B”之间的 LCS 是多少？
# 
# **现在 B 和 B 匹配了，在相应的单元格中填上值 1**。
# 
# <img src='notebook_ims/matrix_3_match.png' width=25% />
# 
# 继续下去，最终矩阵如下所示，在右下角有个 **2**。
# 
# <img src='notebook_ims/matrix_6_complete.png' width=25% />
# 
# 最终的 LCS 等于值 **2** 除以 A 中的 n-gram 数量。所以标准化值为 2/4 = **0.5**。
# 
# ### 矩阵规则
# 
# 要注意的一点是，你可以一次一个单元格地填充该矩阵。每个网格的值仅取决于紧挨着的顶部和左侧网格中的值，或者对角线/左上角的值。规则如下所示：
# * 首先是有一个多余行和列（填充 0）的矩阵。
# * 在遍历字符串时：
#     * 如果有匹配，则用左上角的值加一后填充该单元格。在此示例中，当我们发现匹配的 B-B 时，将匹配单元格左上角的值 0 加 1 后填充到该单元格。
#     * 如果没有匹配，将紧挨着的左侧和上方单元格中的值之最大值填充到非匹配单元格中。
# 
# <img src='notebook_ims/matrix_rules.png' width=50% />
# 
# 填完矩阵后，**右下角的单元格将包含非标准化 LCS 值**。
# 
# 这种矩阵方法可以应用到一组字词上，而非仅仅是字母上。你的函数应该将此矩阵应用到两个文本中的字词上，并返回标准化 LCS 值。

# In[16]:


# Compute the normalized LCS given an answer text and a source text
def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''
    
    a_text = answer_text.split()
    s_text = source_text.split()
    
    lcs_matrix = np.zeros((len(s_text) + 1, len(a_text) + 1), dtype=int)
    
    for i in range(1, len(s_text)+1):
        for j in range(1, len(a_text)+1):
            if s_text[i-1] == a_text[j-1]:
                lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i-1][j], lcs_matrix[i][j-1])
    
    lcs = lcs_matrix[len(s_text)][len(a_text)]
    
    return lcs / len(a_text)


# ### 测试单元格
# 
# 首先用在开头的描述中提供的示例测试你的代码。
# 
# 在以下单元格中，我们指定了字符串 A（答案文本）和 S（原文）。我们知道这两段文本有 20 个公共字词，提交的答案文本长 27，所以标准化的 LCS 应为 20/27。
# 

# In[17]:


# Run the test scenario from above
# does your function return the expected value?

A = "i think pagerank is a link analysis algorithm used by google that uses a system of weights attached to each element of a hyperlinked set of documents"
S = "pagerank is a link analysis algorithm used by the google internet search engine that assigns a numerical weighting to each element of a hyperlinked set of documents"

# calculate LCS
lcs = lcs_norm_word(A, S)
print('LCS = ', lcs)


# expected value test
assert lcs==20/27., "Incorrect LCS value, expected about 0.7408, got "+str(lcs)

print('Test passed!')


# 下个单元格会运行更严格的测试。

# In[18]:


# run test cell
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# test lcs implementation
# params: complete_df from before, and lcs_norm_word function
tests.test_lcs(complete_df, lcs_norm_word)


# 最后，看看 `lcs_norm_word` 生成的几个值。与之前一样，你应该看到更高的值对应于高更级别的剽窃。

# In[19]:


# test on your own
test_indices = range(5) # look at first few files

category_vals = []
lcs_norm_vals = []
# iterate through first few docs and calculate LCS
for i in test_indices:
    category_vals.append(complete_df.loc[i, 'Category'])
    # get texts to compare
    answer_text = complete_df.loc[i, 'Text'] 
    task = complete_df.loc[i, 'Task']
    # we know that source texts have Class = -1
    orig_rows = complete_df[(complete_df['Class'] == -1)]
    orig_row = orig_rows[(orig_rows['Task'] == task)]
    source_text = orig_row['Text'].values[0]
    
    # calculate lcs
    lcs_val = lcs_norm_word(answer_text, source_text)
    lcs_norm_vals.append(lcs_val)

# print out result, does it make sense?
print('Original category values: \n', category_vals)
print()
print('Normalized LCS values: \n', lcs_norm_vals)


# ---
# # 创建所有特征
# 
# 完成了特征计算函数后，下面开始创建多个特征，并判断要在最终模型中使用哪些特征。在以下单元格中，我们提供了两个辅助函数，帮助你创建多个特征并将这些特征存储到 DataFrame `features_df` 中。
# 
# ### 创建多个包含系数特征
# 
# 你完成的 `calculate_containment` 函数将在下个单元格中被调用，该单元格定义了辅助函数 `create_containment_features`。
# 
# > 此函数返回了一个包含系数特征列表，并根据给定的 `n` 和 df（假设为 `complete_df`）中的所有文件计算而出。
# 
# 对于原始文件，包含系数值设为特殊值 -1。
# 
# 你可以通过该函数轻松地针对每个文本文件创建多个包含系数特征，每个特征的 n-gram 长度都不一样。

# In[20]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Function returns a list of containment features, calculated for a given n 
# Should return a list of length 100 for all files in a complete_df
def create_containment_features(df, n, column_name=None):
    
    containment_values = []
    
    if(column_name==None):
        column_name = 'c_'+str(n) # c_1, c_2, .. c_n
    
    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i,'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks 
        else:
            containment_values.append(-1)
    
    print(str(n)+'-gram containment features created!')
    return containment_values


# ### 创建 LCS 特征
# 
# 在以下单元格中，你完成的 `lcs_norm_word` 函数将用于为给定 DataFrame 中的所有答案文件创建一个 LCS 特征列表（同样假设你传入的是 `complete_df`）。它会为原文分配特殊值 -1。
# 

# In[21]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Function creates lcs feature and add it to the dataframe
def create_lcs_features(df, column_name='lcs_word'):
    
    lcs_values = []
    
    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i,'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text'] 
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks 
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_values
    


# ## 练习：通过选择 `ngram_range` 创建特征 DataFrame
# 
# 论文建议计算以下特征：*1-gram 到 5-gram* 包含系数和*最长公共子序列*。 
# > 在这道练习中，你可以选择创建更多的特征，例如 *1-gram 到 7-gram* 包含系数特征和*最长公共子序列*。
# 
# 你需要创建至少 6 个特征，并从中选择一些特征添加到最终的分类模型中。定义和比较至少 6 个不同的特征使你能够丢弃似乎多余的任何特征，并选择用在最终模型中的最佳特征。
# 
# 在以下单元格中，请**定义 n-gram 范围**；你将使用这些 n 创建 n-gram 包含系数特征。我们提供了剩余的特征创建代码。

# In[22]:


# Define an ngram range
ngram_range = range(1,7)


# The following code may take a minute to run, depending on your ngram_range
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
features_list = []

# Create features in a features_df
all_features = np.zeros((len(ngram_range)+1, len(complete_df)))

# Calculate features for containment for ngrams in range
i=0
for n in ngram_range:
    column_name = 'c_'+str(n)
    features_list.append(column_name)
    # create containment features
    all_features[i]=np.squeeze(create_containment_features(complete_df, n))
    i+=1

# Calculate features for LCS_Norm Words 
features_list.append('lcs_word')
all_features[i]= np.squeeze(create_lcs_features(complete_df))

# create a features dataframe
features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

# Print all features/columns
print()
print('Features: ', features_list)
print()


# In[23]:


# print some results 
features_df.head(10)


# ## 相关特征
# 
# 你应该检查整个数据集的特征相关性，判断哪些特征**过于高度相似**，没必要都包含在一个模型中。在分析过程中，你可以使用整个数据集，因为我们的样本量很小。
# 
# 所有特征都尝试衡量两段文本之间的相似性。因为特征都是为了衡量相似性，所以这些特征可能会高度相关。很多分类模型（例如朴素贝叶斯分类器）都要求特征不高度相关；高度相关的特征可能会过于突出单个特征的重要性。
# 
# 所以你在选择特征时，需要选择相关性低的几个特征。相关系数值的范围从 0 到 1，表示从低到高，如以下[相关性矩阵](https://www.displayr.com/what-is-a-correlation-matrix/)所示。

# In[24]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Create correlation matrix for just Features to determine different models to test
corr_matrix = features_df.corr().abs().round(2)

# display shows all of a dataframe
display(corr_matrix)


# ## 练习：创建选择的训练/测试数据
# 
# 请在下方完成函数 `train_test_data`。此函数的参数应该包括：
# * `complete_df`：一个 DataFrame，其中包含所有处理过的文本数据、文件信息、数据类型和类别标签
# * `features_df`：一个 DataFrame，其中包含所有计算的特征，例如 ngram (n= 1-5) 的包含系数，以及 `complete_df`（在之前的单元格中创建的 df）中列出的每个文本文件的 LCS 值。
# * `selected_features`：一个特征列名称列表，例如 `['c_1', 'lcs_word']`，将用于在创建训练/测试数据集时选择最终特征。
# 
# 它应该返回两个元组：
# * `(train_x, train_y)`，所选的训练特征及其对应的类别标签 (0/1)
# * `(test_x, test_y)`，所选的测试特征及其对应的类别标签 (0/1)
# 
# ** 注意：x 和 y 应该分别是特征值和数值类别标签数组，不是 DataFrame。**
# 
# 在看了上述相关性矩阵后，你应该设置一个小于 1.0 的相关性**边界**值，判断哪些特征过于高度相关，不适合包含在最终训练和测试数据中。如果你找不到相关性比边界值更低的特征，建议增加特征数量（更长的 n-gram）并从中选择特征，或者在最终模型中仅使用一两个特征，避免引入高度相关的特征。
# 
# `complete_df` 有一个 `Datatype` 列，表示数据应该为 `train` 或 `test` 数据；它可以帮助你相应地拆分数据。

# In[25]:


# Takes in dataframes and a list of selected features (column names) 
# and returns (train_x, train_y), (test_x, test_y)
def train_test_data(complete_df, features_df, selected_features):
    '''Gets selected training and test features from given dataframes, and 
       returns tuples for training and test features and their corresponding class labels.
       :param complete_df: A dataframe with all of our processed text data, datatypes, and labels
       :param features_df: A dataframe of all computed, similarity features
       :param selected_features: An array of selected features that correspond to certain columns in `features_df`
       :return: training and test features and labels: (train_x, train_y), (test_x, test_y)'''
    
    df = pd.concat([complete_df, features_df[selected_features]], axis=1)    
    df_train = df[df['Datatype'] == 'train']
    df_test = df[df['Datatype'] == 'test']

    # get the training features
    train_x = df_train[selected_features].values
    # And training class labels (0 or 1)
    train_y = df_train['Class'].values
    
    # get the test features and labels
    test_x = df_test[selected_features].values
    test_y = df_test['Class'].values
    
    return (train_x, train_y), (test_x, test_y)


# ### 测试单元格
# 
# 请在下面测试你的实现代码并创建最终训练/测试数据。

# In[26]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
test_selection = list(features_df)[:2] # first couple columns as a test
# test that the correct train/test data is created
(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, test_selection)

# params: generated train/test data
tests.test_data_split(train_x, train_y, test_x, test_y)


# ## 练习：选择“合适的”特征
# 
# 如果你通过了上述测试，就可以在下面创建训练和测试数据集了。
# 
# 定义一个将包含在最终模型中的特征列表 `selected_features`；此列表列出了你想要包含的特征的名称。

# In[27]:


# Select your list of features, this should be column names from features_df
# ex. ['c_1', 'lcs_word']
selected_features = ['c_1', 'c_5', 'lcs_word']


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

(train_x, train_y), (test_x, test_y) = train_test_data(complete_df, features_df, selected_features)

# check that division of samples seems correct
# these should add up to 95 (100 - 5 original files)
print('Training size: ', len(train_x))
print('Test size: ', len(test_x))
print()
print('Training df sample: \n', train_x[:10])


# ### 问题 2：你是如何决定要在最终模型中包含哪些特征的？ 

# **回答：**
# 
# 因为（c_1）与（c_9，c_10，c_11）之间的相关性最小，并且（c_9，c_10，c_11）相关性很大，因此从（c_9，c_10，c_11）选择一个指标，为c_9。最长的公共子序列（lcs_word）能够提供额外的信息，因此我也选择包含了这个指标作为特征。

# ---
# ## 创建最终数据文件
# 
# 现在几乎已经准备好在 SageMaker 中训练模型了。
# 
# 你需要在 SageMaker 中访问训练和测试数据并将其上传到 S3。在此项目中，SageMaker 要求训练/测试数据符合以下格式：
# * 训练数据和测试数据分别保存在一个 `.csv` 文件中，例如 `train.csv` 和 `test.csv`
# * 这些文件的第一列应该是类别标签，其余列是特征。
# 
# 这种格式符合 [SageMaker 文档](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html)的规定：Amazon SageMaker 要求 CSV 文件没有标题，并且目标变量[类别标签]在第一列。
# 
# ## 练习：创建 csv 文件
# 
# 定义一个函数，它会接受 x（特征）和 y（标签），并将它们保存到路径 `data_dir/filename`.下的一个 `.csv` 文件中。
# 
# 建议使用 pandas 将特征和标签合并成一个 DataFrame，并将其转换成 csv 文件。你可以使用 `dropna` 删除 DataFrame 中任何不完整的行。

# In[28]:


def make_csv(x, y, filename, data_dir):
    '''Merges features and labels and converts them into one csv file with labels in the first column.
       :param x: Data features
       :param y: Data labels
       :param file_name: Name of csv file, ex. 'train.csv'
       :param data_dir: The directory where files will be saved
       '''
    # make data dir, if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    
    # your code here
    pd.concat([pd.DataFrame(y), pd.DataFrame(x)], axis=1).to_csv(os.path.join(data_dir, filename), header=False, index=False)
    
    # nothing is returned, but a print statement indicates that the function has run
    print('Path created: '+str(data_dir)+'/'+str(filename))


# ### 测试单元格
# 
# 测试在给定一些文本特征和标签后，代码是否会生成正确格式的 `.csv` 文件。

# In[29]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
fake_x = [ [0.39814815, 0.0001, 0.19178082], 
           [0.86936937, 0.44954128, 0.84649123], 
           [0.44086022, 0., 0.22395833] ]

fake_y = [0, 1, 1]

make_csv(fake_x, fake_y, filename='to_delete.csv', data_dir='test_csv')

# read in and test dimensions
fake_df = pd.read_csv('test_csv/to_delete.csv', header=None)

# check shape
assert fake_df.shape==(3, 4),       'The file should have as many rows as data_points and as many columns as features+1 (for indices).'
# check that first column = labels
assert np.all(fake_df.iloc[:,0].values==fake_y), 'First column is not equal to the labels, fake_y.'
print('Tests passed!')


# In[30]:


# delete the test csv file, generated above
get_ipython().system(' rm -rf test_csv')


# 如果你通过了上述测试，请运行以下单元格以在你指定的目录中创建 `train.csv` 和 `test.csv` 文件。它会将数据保存到本地目录中。请记住该目录的名称，因为在将数据上传到 S3 时需要引用该目录。

# In[31]:


# can change directory, if you want
data_dir = 'plagiarism_data'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

make_csv(train_x, train_y, filename='train.csv', data_dir=data_dir)
make_csv(test_x, test_y, filename='test.csv', data_dir=data_dir)


# ## 后续步骤
# 
# 你已经执行了特征工程并创建了训练和测试数据，接下来可以训练和部署剽窃分类模型了。下个 notebook 将利用 SageMaker 资源训练和测试你设计的模型。

# In[ ]:




