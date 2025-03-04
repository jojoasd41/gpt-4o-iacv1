import json
from sklearn import metrics  # 机器学习评估指标
import pandas as pd


def eval_performance(y_true, y_pred, metric_path=None):
    # 性能评估
    # 计算精度
    metric_dict = {}
    precision = metrics.precision_score(y_true, y_pred)
    print("Precision:\n\t", precision)
    metric_dict['Precision'] = precision

    # Recall
    recall = metrics.recall_score(y_true, y_pred)
    print("Recall:\n\t", recall)
    metric_dict['Recall'] = recall

    # Accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy:\n\t", accuracy)
    metric_dict['Accuracy'] = accuracy

    print("-------------------F1, Micro-F1, Macro-F1, Weighted-F1..-------------------------")
    print("-------------------**********************************-------------------------")

    # F1 Score
    f1 = metrics.f1_score(y_true, y_pred)
    print("F1 Score:\n\t", f1)
    metric_dict['F1'] = f1

    # Micro-F1 Score
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    print("Micro-F1 Score:\n\t", micro_f1)
    metric_dict['Micro-F1'] = micro_f1

    # Macro-F1 Score
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print("Macro-F1 Score:\n\t", macro_f1)
    metric_dict['Macro-F1'] = macro_f1

    # Weighted-F1 Score
    weighted_f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    print("Weighted-F1 Score:\n\t", weighted_f1)
    metric_dict['Weighted-F1'] = weighted_f1

    print("------------------**********************************-------------------------")
    print("-------------------**********************************-------------------------")

    # ROC AUC Score
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        print("ROC AUC:\n\t", roc_auc)
    except:
        print('Only one class present in y_true. ROC AUC score is not defined in that case.')
        metric_dict['ROC-AUC'] = 0

    # Confusion matrix
    print("Confusion Matrix:\n\t", metrics.confusion_matrix(y_true, y_pred))

    if metric_path is not None:
        json.dump(metric_dict, open(metric_path, 'w'), indent=4)


metric_path = '2'

output_path = '1'

# 定义所有分块的 CSV 文件路径（按顺序排列）
chunk_files = []
for i in range(1, 33):
    file_path = f"gpt-4o_io_{i}.csv"
    chunk_files.append(file_path)

# 读取所有分块文件并合并
df_list = [pd.read_csv(file) for file in chunk_files]
df = pd.concat(df_list, ignore_index=True)

df.to_csv(output_path, index=0)  # 将合并后的数据框保存为 CSV 文件

eval_performance(df['Label'], df['pred'], metric_path)  # 调用性能评估函数，计算并保存模型的性能指标
