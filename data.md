
baseline
1. 专用排名指标（漏洞定位核心）
表格
指标名	数值	指标含义
acc@5	0.444	Top5 准确率，衡量前 5 个推荐结果中命中漏洞的比例
MAP@5	0.234	前 5 名平均精确率，衡量前 5 个推荐位置的漏洞识别综合精度
nDCG@5	0.314	前 5 名归一化折损累积增益，衡量漏洞推荐结果的排序质量
MFR	19.816	平均首次检出排名，找到首个真实漏洞平均需检查 19.8 行代码
MAR	23.179	平均绝对排名，所有真实漏洞的平均排名位置为 23.2 行
2. 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
表格
指标名	数值	指标名	数值
func_f1	0.727	func_mcc	0.590
func_rec	0.908	func_fpr	0.275
func_prec	0.606	func_fnr	0.092
func_rocauc	0.864	func_prauc	0.801
（2）语句级（stmt_）漏洞检测指标
表格
指标名	数值	指标名	数值
stmt_f1	0.064	stmt_mcc	0.037
stmt_rec	0.232	stmt_fpr	0.146
stmt_prec	0.037	stmt_fnr	0.768
stmt_rocauc	0.618	stmt_prauc	0.510
stmt_prauc_pos	0.035	-	-
（3）代码行级（stmtline_）漏洞检测指标
表格
指标名	数值	指标名	数值
stmtline_f1	0.058	stmtline_mcc	0.027
stmtline_rec	0.160	stmtline_fpr	0.106
stmtline_prec	0.036	stmtline_fnr	0.840
stmtline_rocauc	0.583	stmtline_prauc	0.508
stmtline_prauc_pos	0.032	-	-
3. 训练损失指标
表格
指标名	数值
stmt_loss	0.3481
func_loss	0.6400
stmtline_loss	0.3512


去掉graphcodebert
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.464，含义：前 5 个推荐结果中命中真实漏洞的比例，衡量模型的 top 推荐精度
MAP@5（前 5 名平均精确率）：0.254，含义：对每个函数计算前 5 个预测结果的平均精确率，再取所有函数的平均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.336，含义：考虑预测位置权重，越靠前的正确预测权重越高，衡量排序质量
MFR（平均故障排名）：20.337，含义：找到第一个真实漏洞所需检查的平均代码行数，数值越低定位效率越高
MAR（平均绝对排名）：23.51，含义：所有真实漏洞的平均排名位置，数值越低漏洞排序越靠前
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.711；func_mcc（马修斯相关系数）：0.566
func_rec（召回率）：0.908；func_fpr（假阳性率）：0.301
func_prec（精确率）：0.585；func_fnr（假阴性率）：0.092
func_rocauc（ROC 曲线下面积）：0.86；func_prauc（PR 曲线下面积）：0.802
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.098；stmt_mcc（马修斯相关系数）：0.074
stmt_rec（召回率）：0.177；stmt_fpr（假阳性率）：0.059
stmt_prec（精确率）：0.068；stmt_fnr（假阴性率）：0.823
stmt_rocauc（ROC 曲线下面积）：0.666；stmt_prauc（PR 曲线下面积）：0.52
stmt_prauc_pos：0.055
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.087；stmtline_mcc（马修斯相关系数）：0.062
stmtline_rec（召回率）：0.179；stmtline_fpr（假阳性率）：0.072
stmtline_prec（精确率）：0.057；stmtline_fnr（假阴性率）：0.821
stmtline_rocauc（ROC 曲线下面积）：0.657；stmtline_prauc（PR 曲线下面积）：0.517
stmtline_prauc_pos：0.047
2.3 训练损失指标
stmt_loss（语句级损失）：0.3921，含义：模型在语句级漏洞分类任务中的训练损失
func_loss（函数级损失）：0.6529，含义：模型在函数级漏洞分类任务中的训练损失
stmtline_loss（代码行级损失）：0.4108，含义：模型在代码行级漏洞分类任务中的训练损失


去掉残差
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.593，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.389，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.471，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：13.712，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：17.106，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.711；func_mcc（马修斯相关系数）：0.565
func_rec（召回率）：0.911；func_fpr（假阳性率）：0.304
func_prec（精确率）：0.583；func_fnr（假阴性率）：0.089
func_rocauc（ROC 曲线下面积）：0.856；func_prauc（PR 曲线下面积）：0.805
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.175；stmt_mcc（马修斯相关系数）：0.154
stmt_rec（召回率）：0.185；stmt_fpr（假阳性率）：0.023
stmt_prec（精确率）：0.166；stmt_fnr（假阴性率）：0.815
stmt_rocauc（ROC 曲线下面积）：0.688；stmt_prauc（PR 曲线下面积）：0.547
stmt_prauc_pos：0.109
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.182；stmtline_mcc（马修斯相关系数）：0.162
stmtline_rec（召回率）：0.185；stmtline_fpr（假阳性率）：0.021
stmtline_prec（精确率）：0.180；stmtline_fnr（假阴性率）：0.815
stmtline_rocauc（ROC 曲线下面积）：0.729；stmtline_prauc（PR 曲线下面积）：0.556
stmtline_prauc_pos：0.122
2.3 训练损失指标
stmt_loss（语句级损失）：0.3738，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6473，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3869，含义：代码行级漏洞分类任务的训练损失

去掉gcn
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.596，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.389，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.479，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：12.968，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：16.341，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.707；func_mcc（马修斯相关系数）：0.558
func_rec（召回率）：0.908；func_fpr（假阳性率）：0.309
func_prec（精确率）：0.578；func_fnr（假阴性率）：0.092
func_rocauc（ROC 曲线下面积）：0.855；func_prauc（PR 曲线下面积）：0.805
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.162；stmt_mcc（马修斯相关系数）：0.141
stmt_rec（召回率）：0.206；stmt_fpr（假阳性率）：0.033
stmt_prec（精确率）：0.134；stmt_fnr（假阴性率）：0.794
stmt_rocauc（ROC 曲线下面积）：0.649；stmt_prauc（PR 曲线下面积）：0.533
stmt_prauc_pos：0.084
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.176；stmtline_mcc（马修斯相关系数）：0.155
stmtline_rec（召回率）：0.209；stmtline_fpr（假阳性率）：0.028
stmtline_prec（精确率）：0.153；stmtline_fnr（假阴性率）：0.791
stmtline_rocauc（ROC 曲线下面积）：0.73；stmtline_prauc（PR 曲线下面积）：0.552
stmtline_prauc_pos：0.115
2.3 训练损失指标
stmt_loss（语句级损失）：0.3701，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6372，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3895，含义：代码行级漏洞分类任务的训练损失

去掉gatv2
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.538，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.349，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.431，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：15.61，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：18.932，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.694；func_mcc（马修斯相关系数）：0.535
func_rec（召回率）：0.888；func_fpr（假阳性率）：0.314
func_prec（精确率）：0.569；func_fnr（假阴性率）：0.112
func_rocauc（ROC 曲线下面积）：0.848；func_prauc（PR 曲线下面积）：0.799
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.132；stmt_mcc（马修斯相关系数）：0.114
stmt_rec（召回率）：0.119；stmt_fpr（假阳性率）：0.017
stmt_prec（精确率）：0.148；stmt_fnr（假阴性率）：0.881
stmt_rocauc（ROC 曲线下面积）：0.654；stmt_prauc（PR 曲线下面积）：0.528
stmt_prauc_pos：0.073
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.154；stmtline_mcc（马修斯相关系数）：0.133
stmtline_rec（召回率）：0.155；stmtline_fpr（假阳性率）：0.021
stmtline_prec（精确率）：0.153；stmtline_fnr（假阴性率）：0.845
stmtline_rocauc（ROC 曲线下面积）：0.745；stmtline_prauc（PR 曲线下面积）：0.544
stmtline_prauc_pos：0.096
2.3 训练损失指标
stmt_loss（语句级损失）：0.3882，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6493，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.4181，含义：代码行级漏洞分类任务的训练损失

去掉focal loss
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.551，即前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.357，各函数前 5 个预测结果的平均精确率均值，反映漏洞排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.441，考虑预测位置权重，评估漏洞排序质量
MFR（平均故障排名）：15.489，找到第一个真实漏洞所需检查的平均代码行数
MAR（平均绝对排名）：18.952，所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.691；func_mcc（马修斯相关系数）：0.529
func_rec（召回率）：0.881；func_fpr（假阳性率）：0.312
func_prec（精确率）：0.568；func_fnr（假阴性率）：0.119
func_rocauc（ROC 曲线下面积）：0.847；func_prauc（PR 曲线下面积）：0.795
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.137；stmt_mcc（马修斯相关系数）：0.115
stmt_rec（召回率）：0.156；stmt_fpr（假阳性率）：0.027
stmt_prec（精确率）：0.123；stmt_fnr（假阴性率）：0.844
stmt_rocauc（ROC 曲线下面积）：0.651；stmt_prauc（PR 曲线下面积）：0.528
stmt_prauc_pos：0.074
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.163；stmtline_mcc（马修斯相关系数）：0.142
stmtline_rec（召回率）：0.168；stmtline_fpr（假阳性率）：0.022
stmtline_prec（精确率）：0.159；stmtline_fnr（假阴性率）：0.832
stmtline_rocauc（ROC 曲线下面积）：0.757；stmtline_prauc（PR 曲线下面积）：0.545
stmtline_prauc_pos：0.099
2.3 训练损失指标
stmt_loss（语句级损失）：0.3450，模型在语句级漏洞分类任务中的训练损失
func_loss（函数级损失）：0.6106，模型在函数级漏洞分类任务中的训练损失
stmtline_loss（代码行级损失）：0.3495，模型在代码行级漏洞分类任务中的训练损失



ours1

acc@5（Top5 准确率）：0.603，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.402，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.492，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：13.256，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：16.751，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.704；func_mcc（马修斯相关系数）：0.556
func_rec（召回率）：0.918；func_fpr（假阳性率）：0.322
func_prec（精确率）：0.571；func_fnr（假阴性率）：0.082
func_rocauc（ROC 曲线下面积）：0.855；func_prauc（PR 曲线下面积）：0.805
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.149；stmt_mcc（马修斯相关系数）：0.127
stmt_rec（召回率）：0.202；stmt_fpr（假阳性率）：0.037
stmt_prec（精确率）：0.118；stmt_fnr（假阴性率）：0.798
stmt_rocauc（ROC 曲线下面积）：0.655；stmt_prauc（PR 曲线下面积）：0.535
stmt_prauc_pos：0.087
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.202；stmtline_mcc（马修斯相关系数）：0.182
stmtline_rec（召回率）：0.214；stmtline_fpr（假阳性率）：0.022
stmtline_prec（精确率）：0.192；stmtline_fnr（假阴性率）：0.786
stmtline_rocauc（ROC 曲线下面积）：0.744；stmtline_prauc（PR 曲线下面积）：0.563
stmtline_prauc_pos：0.135
2.3 训练损失指标
stmt_loss（语句级损失）：0.3682，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6385，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3859，含义：代码行级漏洞分类任务的训练损失
## 主实验

ours1

acc@5（Top5 准确率）：0.603，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.402，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.492，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：13.256，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：16.751，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.704；func_mcc（马修斯相关系数）：0.556
func_rec（召回率）：0.918；func_fpr（假阳性率）：0.322
func_prec（精确率）：0.571；func_fnr（假阴性率）：0.082
func_rocauc（ROC 曲线下面积）：0.855；func_prauc（PR 曲线下面积）：0.805
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.149；stmt_mcc（马修斯相关系数）：0.127
stmt_rec（召回率）：0.202；stmt_fpr（假阳性率）：0.037
stmt_prec（精确率）：0.118；stmt_fnr（假阴性率）：0.798
stmt_rocauc（ROC 曲线下面积）：0.655；stmt_prauc（PR 曲线下面积）：0.535
stmt_prauc_pos：0.087
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.202；stmtline_mcc（马修斯相关系数）：0.182
stmtline_rec（召回率）：0.214；stmtline_fpr（假阳性率）：0.022
stmtline_prec（精确率）：0.192；stmtline_fnr（假阴性率）：0.786
stmtline_rocauc（ROC 曲线下面积）：0.744；stmtline_prauc（PR 曲线下面积）：0.563
stmtline_prauc_pos：0.135
2.3 训练损失指标
stmt_loss（语句级损失）：0.3682，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6385，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3859，含义：代码行级漏洞分类任务的训练损失

ours2
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.588，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.401，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.483，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：13.792，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：16.764，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.704；func_mcc（马修斯相关系数）：0.557
func_rec（召回率）：0.926；func_fpr（假阳性率）：0.328
func_prec（精确率）：0.569；func_fnr（假阴性率）：0.074
func_rocauc（ROC 曲线下面积）：0.855；func_prauc（PR 曲线下面积）：0.804
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.159；stmt_mcc（马修斯相关系数）：0.137
stmt_rec（召回率）：0.208；stmt_fpr（假阳性率）：0.035
stmt_prec（精确率）：0.128；stmt_fnr（假阴性率）：0.792
stmt_rocauc（ROC 曲线下面积）：0.641；stmt_prauc（PR 曲线下面积）：0.534
stmt_prauc_pos：0.086
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.189；stmtline_mcc（马修斯相关系数）：0.170
stmtline_rec（召回率）：0.179；stmtline_fpr（假阳性率）：0.018
stmtline_prec（精确率）：0.199；stmtline_fnr（假阴性率）：0.821
stmtline_rocauc（ROC 曲线下面积）：0.754；stmtline_prauc（PR 曲线下面积）：0.562
stmtline_prauc_pos：0.133
2.3 训练损失指标
stmt_loss（语句级损失）：0.3672，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6363，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3937，含义：代码行级漏洞分类任务的训练损失

ours3
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.566，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.390，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.468，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：15.499，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：19.191，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.707；func_mcc（马修斯相关系数）：0.558
func_rec（召回率）：0.906；func_fpr（假阳性率）：0.307
func_prec（精确率）：0.579；func_fnr（假阴性率）：0.094
func_rocauc（ROC 曲线下面积）：0.855；func_prauc（PR 曲线下面积）：0.805
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.151；stmt_mcc（马修斯相关系数）：0.143
stmt_rec（召回率）：0.116；stmt_fpr（假阳性率）：0.010
stmt_prec（精确率）：0.215；stmt_fnr（假阴性率）：0.884
stmt_rocauc（ROC 曲线下面积）：0.627；stmt_prauc（PR 曲线下面积）：0.534
stmt_prauc_pos：0.086
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.206；stmtline_mcc（马修斯相关系数）：0.186
stmtline_rec（召回率）：0.211；stmtline_fpr（假阳性率）：0.020
stmtline_prec（精确率）：0.201；stmtline_fnr（假阴性率）：0.789
stmtline_rocauc（ROC 曲线下面积）：0.711；stmtline_prauc（PR 曲线下面积）：0.553
stmtline_prauc_pos：0.117
2.3 训练损失指标
stmt_loss（语句级损失）：0.3548，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6362，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3700，含义：代码行级漏洞分类任务的训练损失

ours4
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.548，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.354，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.441，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：15.084，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：18.378，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.695；func_mcc（马修斯相关系数）：0.536
func_rec（召回率）：0.876；func_fpr（假阳性率）：0.301
func_prec（精确率）：0.576；func_fnr（假阴性率）：0.124
func_rocauc（ROC 曲线下面积）：0.849；func_prauc（PR 曲线下面积）：0.799
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.137；stmt_mcc（马修斯相关系数）：0.116
stmt_rec（召回率）：0.200；stmt_fpr（假阳性率）：0.042
stmt_prec（精确率）：0.105；stmt_fnr（假阴性率）：0.800
stmt_rocauc（ROC 曲线下面积）：0.659；stmt_prauc（PR 曲线下面积）：0.536
stmt_prauc_pos：0.088
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.160；stmtline_mcc（马修斯相关系数）：0.154
stmtline_rec（召回率）：0.122；stmtline_fpr（假阳性率）：0.010
stmtline_prec（精确率）：0.232；stmtline_fnr（假阴性率）：0.878
stmtline_rocauc（ROC 曲线下面积）：0.755；stmtline_prauc（PR 曲线下面积）：0.551
stmtline_prauc_pos：0.111
2.3 训练损失指标
stmt_loss（语句级损失）：0.3788，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6512，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3997，含义：代码行级漏洞分类任务的训练损失



## 对比实验

codebert
2.1 专用排名指标（漏洞定位核心）
acc@5（Top5 准确率）：0.464，含义：前 5 个推荐结果中命中真实漏洞的比例，衡量模型的 top 推荐精度
MAP@5（前 5 名平均精确率）：0.254，含义：对每个函数计算前 5 个预测结果的平均精确率，再取所有函数的平均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.336，含义：考虑预测位置权重，越靠前的正确预测权重越高，衡量排序质量
MFR（平均故障排名）：20.337，含义：找到第一个真实漏洞所需检查的平均代码行数，数值越低定位效率越高
MAR（平均绝对排名）：23.51，含义：所有真实漏洞的平均排名位置，数值越低漏洞排序越靠前
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.711；func_mcc（马修斯相关系数）：0.566
func_rec（召回率）：0.908；func_fpr（假阳性率）：0.301
func_prec（精确率）：0.585；func_fnr（假阴性率）：0.092
func_rocauc（ROC 曲线下面积）：0.86；func_prauc（PR 曲线下面积）：0.802
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.098；stmt_mcc（马修斯相关系数）：0.074
stmt_rec（召回率）：0.177；stmt_fpr（假阳性率）：0.059
stmt_prec（精确率）：0.068；stmt_fnr（假阴性率）：0.823
stmt_rocauc（ROC 曲线下面积）：0.666；stmt_prauc（PR 曲线下面积）：0.52
stmt_prauc_pos：0.055
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.087；stmtline_mcc（马修斯相关系数）：0.062
stmtline_rec（召回率）：0.179；stmtline_fpr（假阳性率）：0.072
stmtline_prec（精确率）：0.057；stmtline_fnr（假阴性率）：0.821
stmtline_rocauc（ROC 曲线下面积）：0.657；stmtline_prauc（PR 曲线下面积）：0.517
stmtline_prauc_pos：0.047
2.3 训练损失指标
stmt_loss（语句级损失）：0.3921，含义：模型在语句级漏洞分类任务中的训练损失
func_loss（函数级损失）：0.6529，含义：模型在函数级漏洞分类任务中的训练损失
stmtline_loss（代码行级损失）：0.4108，含义：模型在代码行级漏洞分类任务中的训练损失


linevd
1. 专用排名指标（漏洞定位核心）
表格
指标名	数值	指标含义
acc@5	0.444	Top5 准确率，衡量前 5 个推荐结果中命中漏洞的比例
MAP@5	0.234	前 5 名平均精确率，衡量前 5 个推荐位置的漏洞识别综合精度
nDCG@5	0.314	前 5 名归一化折损累积增益，衡量漏洞推荐结果的排序质量
MFR	19.816	平均首次检出排名，找到首个真实漏洞平均需检查 19.8 行代码
MAR	23.179	平均绝对排名，所有真实漏洞的平均排名位置为 23.2 行
2. 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
表格
指标名	数值	指标名	数值
func_f1	0.727	func_mcc	0.590
func_rec	0.908	func_fpr	0.275
func_prec	0.606	func_fnr	0.092
func_rocauc	0.864	func_prauc	0.801
（2）语句级（stmt_）漏洞检测指标
表格
指标名	数值	指标名	数值
stmt_f1	0.064	stmt_mcc	0.037
stmt_rec	0.232	stmt_fpr	0.146
stmt_prec	0.037	stmt_fnr	0.768
stmt_rocauc	0.618	stmt_prauc	0.510
stmt_prauc_pos	0.035	-	-
（3）代码行级（stmtline_）漏洞检测指标
表格
指标名	数值	指标名	数值
stmtline_f1	0.058	stmtline_mcc	0.027
stmtline_rec	0.160	stmtline_fpr	0.106
stmtline_prec	0.036	stmtline_fnr	0.840
stmtline_rocauc	0.583	stmtline_prauc	0.508
stmtline_prauc_pos	0.032	-	-
3. 训练损失指标
表格
指标名	数值
stmt_loss	0.3481
func_loss	0.6400
stmtline_loss	0.3512


ours1

acc@5（Top5 准确率）：0.603，含义：前 5 个推荐结果中命中真实漏洞的比例
MAP@5（前 5 名平均精确率）：0.402，含义：各函数前 5 个预测结果的平均精确率均值，反映排序精度
nDCG@5（前 5 名归一化折损累积增益）：0.492，含义：考虑预测位置权重的排序质量评估指标
MFR（平均故障排名）：13.256，含义：找到第一个真实漏洞需检查的平均代码行数
MAR（平均绝对排名）：16.751，含义：所有真实漏洞的平均排名位置
2.2 传统分类指标（分三个粒度）
（1）函数级（func_）漏洞检测指标
func_f1（F1 分数）：0.704；func_mcc（马修斯相关系数）：0.556
func_rec（召回率）：0.918；func_fpr（假阳性率）：0.322
func_prec（精确率）：0.571；func_fnr（假阴性率）：0.082
func_rocauc（ROC 曲线下面积）：0.855；func_prauc（PR 曲线下面积）：0.805
（2）语句级（stmt_）漏洞检测指标
stmt_f1（F1 分数）：0.149；stmt_mcc（马修斯相关系数）：0.127
stmt_rec（召回率）：0.202；stmt_fpr（假阳性率）：0.037
stmt_prec（精确率）：0.118；stmt_fnr（假阴性率）：0.798
stmt_rocauc（ROC 曲线下面积）：0.655；stmt_prauc（PR 曲线下面积）：0.535
stmt_prauc_pos：0.087
（3）代码行级（stmtline_）漏洞检测指标
stmtline_f1（F1 分数）：0.202；stmtline_mcc（马修斯相关系数）：0.182
stmtline_rec（召回率）：0.214；stmtline_fpr（假阳性率）：0.022
stmtline_prec（精确率）：0.192；stmtline_fnr（假阴性率）：0.786
stmtline_rocauc（ROC 曲线下面积）：0.744；stmtline_prauc（PR 曲线下面积）：0.563
stmtline_prauc_pos：0.135
2.3 训练损失指标
stmt_loss（语句级损失）：0.3682，含义：语句级漏洞分类任务的训练损失
func_loss（函数级损失）：0.6385，含义：函数级漏洞分类任务的训练损失
stmtline_loss（代码行级损失）：0.3859，含义：代码行级漏洞分类任务的训练损失



ivdetect

=== 方法级指标 (res2f) ===
准确率: 0.7151
精确率: 0.5320
召回率: 0.8660
F1值: 0.6591
AUC: 0.8147

=== 语句级指标 (res2) ===
准确率: 0.8859
精确率: 0.0475
召回率: 0.1984
F1值: 0.0766
AUC: 0.5991
PR-AUC: 0.5098
MCC: 0.0515
FPR: 0.0973
FNR: 0.8016

排名指标: nDCG@1: 0.000 | MAP@1: 0.000 | FR@1: nan | AR@1: nan | nDCG@3: 0.631 | MAP@3: 0.333 | FR@3: 3.000 | AR@3: 3.000 | nDCG@5: 0.565 | MAP@5: 0.417 | FR@5: 3.000 | AR@5: 3.500 | nDCG@10: 0.565 | MAP@10: 0.417 | FR@10: 3.000 | AR@10: 3.500 | nDCG@10.417 | FR@10: 3.000 | AR@10: 3.500 | nDCG@15: 0.530 | MAP@10.417 | FR@10: 3.000 | AR@10: 3.500 | nDCG@15: 0.530 | MAP@15: 0.349 | FR@15: 3.000 | AR@15: 7.000 | nDCG@20: 0.530 | MAP@20: 0.349 | FR@20: 3.000 | AR@20: 7.000 | AUC: 0.816 | MFR: 3.000 | MAR: 79.967 |     
