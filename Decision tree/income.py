import C45
import treePlotter
import datetime

def split_test_train(dataset,p):  # 按照1：p-1分测试集
    n = len(dataset)
    fore = int(n*p)
    rear = n-fore
    test_data = dataset[:fore]  # 少的部分
    training_data = dataset[-rear:]
    return test_data, training_data

starttime = datetime.datetime.now()
# 读取数据文件
fr = open(r'D:\Download\C4.5\C4.5决策树\train3333.csv')
# 生成数据集
list_n = [inst.strip().split(',') for inst in fr.readlines()]
# 样本特征标签
labels = list_n[0]
lDataSet=list_n[1:]
test_data,training_data=split_test_train(lDataSet, 0.2)
validation_data,training_data=split_test_train(training_data,0.3)
# 样本特征类型，0为离散，1为连续
labelProperties = [1,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1]
#labelProperties = [1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1]
# 类别向量
classList = ['1', '-1']
# 构建决策树
trees = C45.createTree(training_data, labels, labelProperties)
# 绘制决策树
treePlotter.createPlot(trees)
C45.acctesting(trees,classList,test_data,labels,labelProperties)
# 利用验证集对决策树剪枝
C45.postPruningTree(trees, classList, training_data, validation_data, labels, labelProperties)
# 绘制剪枝后的决策树
treePlotter.createPlot(trees)
#print(classLabel)
C45.acctesting(trees,classList,test_data,labels,labelProperties)

endtime = datetime.datetime.now()
print (endtime - starttime)

