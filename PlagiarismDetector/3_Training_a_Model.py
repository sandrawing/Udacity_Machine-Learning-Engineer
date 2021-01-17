#!/usr/bin/env python
# coding: utf-8

# # 剽窃检测模型
# 
# 创建了训练和测试数据后，可以定义和训练模型了。在此 notebook 中，你的目标是训练一个二元分类模型，它会根据你提供的特征学习将答案文件标为剽窃文件或非剽窃文件。
# 
# 此任务将分成以下几个步骤：
# 
# * 将数据上传到 S3。
# * 定义一个二元分类模型和训练脚本。
# * 训练和部署模型。
# * 评估部署的分类器并回答关于所采用方法的一些问题。
# 
# 要完成此 notebook，你需要完成此 notebook 中的所有练习并回答所有问题。
# > 所有任务将清晰地标为**练习**，问题都标为**问题**。
# 
# 你可以尝试不同的分类模型，并选择一个在此数据集上效果最佳的模型。
# 
# ---

# ## 将数据上传到 S3
# 
# 在上个 notebook 中，你应该使用给定剽窃/非剽窃文本数据语料库的特征和类别标签创建了两个文件：`training.csv` 和 `test.csv` 文件。
# 
# >以下单元格将加载一些 AWS SageMaker 库并创建一个默认存储桶。创建此存储桶后，你可以将本地存储的数据上传到 S3。
# 
# 将训练和测试 `.csv` 特征文件保存到本地。你可以在 SageMaker 中运行第二个 notebook“2_Plagiarism_Feature_Engineering”，或者使用 Jupyter Lab 中的上传图标手动将文件上传到此 notebook。然后，你可以使用 `sagemaker_session.upload_data` 将本地文件上传到 S3，并直接指向训练数据的存储位置。

# In[1]:


import pandas as pd
import boto3
import sagemaker


# In[33]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# create an S3 bucket
bucket = sagemaker_session.default_bucket()


# ## 练习：将训练数据上传到 S3
# 
# 指定在其中保存了 `train.csv` 文件的 `data_dir`。指定一个描述性 `prefix`，指出数据将上传到默认 S3 存储桶的什么位置。最后，通过调用 `sagemaker_session.upload_data` 并传入必要的参数，创建一个指向训练数据的指针。建议参考 [Session 文档](https://sagemaker.readthedocs.io/en/stable/session.html#sagemaker.session.Session.upload_data)或之前的 SageMaker 代码示例。
# 
# 你需要上传整个目录。之后，训练脚本将仅访问 `train.csv` 文件。

# In[34]:


# should be the name of directory you created to save your features data
data_dir =  'plagiarism_data'

# set prefix, a descriptive name for a directory  
prefix = 'plagiarism_detection'

# upload all data to S3
input_data = sagemaker_session.upload_data(path= data_dir, bucket=bucket, key_prefix = prefix)


# ### 测试单元格
# 
# 测试数据是否已成功上传。以下单元格将输出 S3 存储桶中的内容，如果为空，将抛出错误。你应该看到 `data_dir` 的内容，或许还有一些检查点。如果你看到其中列出了任何其他文件，那么你也许有一些旧的模型文件，你可以通过 S3 控制台删除这些旧文件（不过多余的文件应该不会影响在此 notebook 中开发的模型的性能）。

# In[35]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# confirm that data is in S3 bucket
empty_check = []
for obj in boto3.resource('s3').Bucket(bucket).objects.all():
    empty_check.append(obj.key)
    print(obj.key)

assert len(empty_check) !=0, 'S3 bucket is empty.'
print('Test passed!')


# ---
# 
# # 建模
# 
# 上传训练数据后，下面定义并训练模型。
# 
# 你可以决定创建什么类型的模型。对于二元分类任务，你可以选择采用以下三种方法之一：
# * 使用内置的分类算法，例如 LinearLearner。
# * 定义自定义 Scikit-learn 分类器，可以在[此处](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)找到各个模型的比较情况。
# * 定义自定义 PyTorch 神经网络分类器。
# 
# 你需要测试各种模型并选择最佳模型。我们将根据最终模型的准确率对你的项目打分。
#  
# ---
# 
# ## 练习：完成训练脚本
# 
# 为了实现自定义分类器，你需要完成 `train.py` 脚本。我们提供了文件夹 `source_sklearn` 和 `source_pytorch`，其中分别包含自定义 Scikit-learn 模型和 PyTorch 模型的起始代码。每个目录都有一个 `train.py` 训练脚本。要完成此项目，**你只需完成其中一个脚本**，即负责训练最终模型的脚本。
# 
# 典型的训练脚本会：
# * 从指定的目录加载训练数据
# * 解析所有的训练和模型超参数（例如神经网络中的节点数，训练周期，等等）
# * 实例化你设计的模型，并采用指定的超参数
# * 训练该模型
# * 最后，保存模型，以便之后托管/部署模型
# 
# ### 定义和训练模型
# 我们已经提供了大部分训练脚本。几乎所有任务都位于 `if __name__ == '__main__':` 部分。为了完成 `train.py` 文件，你需要：
# 1. 导入所需的任何额外库
# 2. 使用 `parser.add_argument` 定义任何其他模型训练超参数
# 2. 在 `if __name__ == '__main__':` 部分定义模型
# 3. 在此部分训练模型
# 
# 你可以在下面使用 `!pygmentize` 显示现有的 `train.py` 文件。请通读代码，所有任务都标有 `TODO` 注释。 
# 
# **注意：如果你选择创建自定义 PyTorch 模型，需要在 `model.py` 文件中定义模型**，并且我们提供了 `predict.py` 文件。如果你选择使用 Scikit-learn，则只需 `train.py` 文件；你可以从 `sklearn` 库中导入一个分类器。

# In[36]:


# directory can be changed to: source_sklearn or source_pytorch
get_ipython().system('pygmentize source_sklearn/train.py')


# ### 提供的代码
# 
# 如果你阅读了上述代码，就会发现起始代码包含：
# * 模型加载 (`model_fn`) 和保存代码
# * 获取 SageMaker 的默认超参数
# * 按照名称 `train.csv` 加载训练数据，并提取特征和标签 `train_x` 和 `train_y`
# 
# 如果你想详细了解如何通过 [joblib for sklearn](https://scikit-learn.org/stable/modules/model_persistence.html) 或 [torch.save](https://pytorch.org/tutorials/beginner/saving_loading_models.html) 保存模型，请点击提供的链接。

# ---
# # 创建评估器
# 
# 在 SageMaker 中构建自定义模型时，必须指定入口点。入口点是一个 Python 文件，当模型被训练时，该文件将执行，即你在上面指定的 `train.py` 函数。要在 SageMaker 中运行自定义训练脚本，你需要构建评估器并指定相应的构造函数参数：
# 
# * **entry_point**：SageMaker 训练模型和预测时运行的 Python 脚本的路径。
# * **source_dir**：训练脚本目录 `source_sklearn` 或 `source_pytorch` 的路径。
# * **entry_point**：SageMaker 训练模型和预测时运行的 Python 脚本的路径。
# * **source_dir**：训练脚本目录 `train_sklearn` 或 `train_pytorch` 的路径。
# * **entry_point**：SageMaker 训练模型时运行的 Python 脚本的路径。
# * **source_dir**：训练脚本目录 `train_sklearn` 或 `train_pytorch` 的路径。
# * **role**：角色 ARN，在上面已指定。
# * **train_instance_count**：训练实例的数量（应该保留为 1）。
# * **train_instance_type**：SageMaker 训练实例的类型。注意，因为 Scikit-learn 不提供 GPU 训练原生支持，所以 Sagemaker Scikit-learn 目前不支持在 GPU 实例上训练模型。
# * **sagemaker_session**：在 Sagemaker 中训练时使用的会话。
# * **hyperparameters**（可选）：作为超参数传递给训练函数的字典 `{'name':value, ..}`。
# 
# 注意：对于 PyTorch 模型，还有一个可选参数 **framework_version**，你可以将其设为最新的 PyTorch 版本 `1.0`。
# 
# ## 练习：定义 Scikit-learn 或 PyTorch 评估器
# 
# 你可以使用以下命令之一导入一个评估器：
# ```
# from sagemaker.sklearn.estimator import SKLearn
# ```
# ```
# from sagemaker.pytorch import PyTorch
# ```

# In[37]:



# your import and estimator code, here
from sagemaker.sklearn.estimator import SKLearn

estimator = SKLearn(entry_point="train.py",
                    source_dir="source_sklearn",
                    role=role,
                    train_instance_count=1,
                    train_instance_type='ml.c4.xlarge')


# ## 练习：训练评估器
# 
# 使用在 S3 中存储的训练数据训练评估器。代码应该创建一个训练作业，你可以在 SageMaker 控制台中监控该作业。

# In[38]:


get_ipython().run_cell_magic('time', '', "\n# Train your estimator on S3 training data\n\nestimator.fit({'train': input_data})")


# ## 练习：部署训练过的模型
# 
# 训练之后，部署模型以创建 `predictor`。如果你使用的是 PyTorch 模型，你需要创建一个训练过的 `PyTorchModel`，它会接受训练过的 `<model>.model_data` 作为输入参数，并指向提供的 `source_pytorch/predict.py` 文件作为入口点。
# 
# 为了部署训练过的模型，你需要使用 `<model>.deploy`，它接受两个参数：
# * **initial_instance_count**：部署实例的数量 (1)。
# * **instance_type**：部署 SageMaker 实例的类型。
# 
# 注意：如果你遇到实例错误，可能是因为你选择了错误的训练或部署实例类型。建议参考之前的练习代码，看看我们使用了哪种类型的实例。

# In[39]:


get_ipython().run_cell_magic('time', '', "\n# uncomment, if needed\n# from sagemaker.pytorch import PyTorchModel\n\n\n# deploy your model to create a predictor\npredictor = estimator.deploy(initial_instance_count=1, instance_type='ml.t2.medium')")


# ---
# # 评估模型
# 
# 模型部署后，你可以将模型应用到测试数据上，看看模型的效果如何。
# 
# 下面提供的单元格会读入测试数据，并假设它存储在本地 `data_dir` 目录下，名称为 `test.csv`。标签和特征是从 `.csv` 文件提取的。

# In[40]:


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import os

# read in test data, assuming it is stored locally
test_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, names=None)

# labels are in the first column
test_y = test_data.iloc[:,0]
test_x = test_data.iloc[:,1:]


# ## 练习：确定模型的准确率
# 
# 使用部署的 `predictor` 为测试数据预测类别标签。将这些标签与真实标签 `test_y` 进行比较，并计算 0-1 之间的准确率，表示模型分类正确的测试数据所占的比例。你可以使用 [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) 计算准确率。
# 
# **要通过此项目，你的模型测试准确率应至少达到 90%。**

# In[41]:


# First: generate predicted, class labels
test_y_preds = predictor.predict(test_x)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# test that your model generates the correct number of labels
assert len(test_y_preds)==len(test_y), 'Unexpected number of predictions.'
print('Test passed!')


# In[42]:


from sklearn.metrics import accuracy_score

# Second: calculate the test accuracy
accuracy = accuracy_score(test_y, test_y_preds)

print(accuracy)


## print out the array of predicted and true labels, if you want
print('\nPredicted class labels: ')
print(test_y_preds)
print('\nTrue class labels: ')
print(test_y.values)


# ### 问题 1：你的模型生成了多少个假正例和假负例？为何会这样？

# **回答：**
# 
# 假正例（False Positive）指的是预测为1，实际为0的样本；假负例（False Negative）指的是预测为0，实际为1的样本。我的模型生成了0个假正例，0个假负例。这可能是因为数据量较小，模型能够很好的拟合。

# ### 问题 2：你是如何决定要使用什么类型的模型？

# **回答：**
# 
# 因为这个是一个分类模型，因此可以选择决策树，支持向量机，逻辑回归等模型。在这里数据集很小，因此决定采用支持向量机分类。

# ----
# ## 练习：清理资源
# 
# 评估完模型后，记得**删除模型端点**。你可以通过调用 `.delete_endpoint()` 删除端点。你需要在此 notebook 中演示端点已删除。你可以从 AWS 控制台删除任何其他资源，并且在下面找到更多关于删除所有资源的说明。

# In[43]:


# uncomment and fill in the line below!
# <name_of_deployed_predictor>.delete_endpoint()

predictor.delete_endpoint()


# ### 删除 S3 存储桶
# 
# 完全训练和测试完模型后，你可以删除整个 S3 存储桶。如果你在训练模型之前删除存储桶，需要重新创建 S3 存储桶并将训练数据再次上传到存储桶中。

# In[44]:


# deleting bucket, uncomment lines below

bucket_to_delete = boto3.resource('s3').Bucket(bucket)
bucket_to_delete.objects.all().delete()


# ### 删除所有模型和实例
# 
# 当你完全处理完模型，并且**不**需要重新访问此 notebook，你可以根据[这些说明](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html)删除所有 SageMaker notebook 实例和模型。在删除此 notebook 实例之前，建议至少下载一个副本并将其保存到本地。

# ---
# ## 后续改进建议
# 
# 此项目还有很多改进或扩展方式，可以拓宽你的知识面，或让此项目更独特。下面是一些建议：
# * 训练分类器预测剽窃类别 (1-3)，而不仅仅是剽窃 (1) 或非剽窃 (0)。
# * 利用其他更大型的数据集检测此模型能否扩展到其他类型的剽窃行为。
# * 利用语言或字符级分析寻找不同（及更多）相似性特征。
# * 编写完整的管道函数，它会接受原文和提交的文本，并将提交的文本分类为剽窃或非剽窃文本。
# * 使用 API Gateway 和 lambda 函数将模型部署到网络应用上。
# 
# 这些都只是扩展项目的建议。如果你完成了此 notebook 中的所有练习，你已经完成了一个真实的应用，可以提交项目了。棒棒哒！
