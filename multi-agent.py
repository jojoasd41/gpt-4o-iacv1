# 结构化标签
import backoff  #
import openai  # OpenAI API接口库
import pandas as pd  # 数据处理库
import json  # JSON格式处理
import os  # 文件系统操作
from tqdm import tqdm  # 进度条显示
import numpy as np  # 数学计算库
from sklearn import metrics  # 机器学习评估指标
from httpx import TimeoutException, TooManyRedirects  # 关键修正点
from openai import OpenAIError  # 确保 OpenAIError 是合法的异常类


@backoff.on_exception(
    backoff.expo,
    (OpenAIError, ConnectionError, TimeoutException, TooManyRedirects),  # 正确异常类
    max_tries=5,
    jitter=backoff.full_jitter
)
def Invoke_model(model, content):  # 调用gpt-4o模型函数，返回响应内容
    openai.api_key = 'sk-ZTY0MDcxNWI1Zjk1ZDc4YmNjNjYyZjZmNjczMDEwOGU'
    openai.api_base = 'https://api.gt4.pro/v1'

    chat_completion = openai.ChatCompletion.create(
        messages=[{"role": "user", "content": content}],
        max_tokens=256,
        model=model,
        stream=False
    )

    # 解析响应内容
    response_content = chat_completion.choices[0].message.content

    return response_content


def generate_CoC_prompt(data_point):  # 只是写起，没有调用过
    """生成认知链（CoC）策略的提示模板"""
    if data_point:
        return f"""
        ### Instruction:
        You are a sarcasm classification classifier. Assign a correct label of the Input text from ['Not Sarcastic', 'Sarcastic'].

        ### Input:
        {data_point}

        You can choose to output the result directly if you believe your judgment is reliable,
        or
        You think step by step if your confidence in your judgment is less than 90%:
        Step 1: What is the SURFACE sentiment, as indicated by clues such as keywords, sentimental phrases, emojis?
        Step 2: Deduce what the sentence really means, namely the TRUE intention, by carefully checking any rhetorical devices, language style, etc.
        Step 3: Compare and analysis Step 1 and Step 2, infer the final sarcasm label.

        ### Response:

        ### Label: 

        """


def Contextual_understanding_prompt(data_point):  # 生成语境分析模板
    if data_point:
        return f""" ### Instruction: You are a classifier for satirical works. Analyze the sentence according to the 
        following steps .Here, [0] indicates this sentence is not a sarcastic expression, [1] indicates This sentence 
        is a sarcastic expression. 

        ### Input Text:
        {data_point}

        ### Steps: 
        1.Responsible for extracting and integrating explicit and implicit contextual information of the 
        text, not only within the context of the text itself, but also actively obtaining contextual information from 
        external knowledge sources. 
        2. Final Label: Brief summary (no more than 10 words!),and Provide the label '[0]' or '[1]', and the answer can only be given 
        with '[0]' or '[1]'. 



        ### Response:
        """


def Semantic_analysis_prompt(data_point):  # 生成语义分析模板
    if data_point:
        return f""" ### Instruction: You are a classifier for satirical works. Analyze the sentence according to the 
        following steps .Here, [0] indicates this sentence is not a sarcastic expression, [1] indicates This sentence 
        is a sarcastic expression. 

        ### Input Text:
        {data_point}

        ### Steps: 
        1.Be responsible for conducting in-depth semantic understanding and sentiment analysis, conducting 
        more granular sentiment dimension analysis (such as sentiment intensity, sentiment type, sentiment 
        orientation), and paying attention to sentiment contrast and inconsistency. 
        2.Final Label: Brief summary (no more than 10 words!),Provide the label '[0]' or '[1]', and the answer can only be given with '[0]' or '[1]'.



        ### Response:
        """


def Rhetorical_Analysis_prompt(data_point):  # 生成修辞分析模板
    if data_point:
        return f""" ### Instruction: You are a classifier for satirical works. Analyze the sentence according to the 
        following steps .Here, [0] indicates this sentence is not a sarcastic expression, [1] indicates This sentence 
        is a sarcastic expression. 

           ### Input:
           {data_point} 

           1.Focus on identifying the rhetorical devices and language techniques used in the text. Not only should one 
           recognize common rhetorical figures (irony, hyperbole, etc.), but also pay attention to the obscure and 
           indirect rhetorical devices.
           2.Final Label: Brief summary (no more than 10 words!),Provide the label '[0]' or '[1]', and the answer can only be given with '[0]' or '[1]'. 






           ### Response:



           """


def Reasoning_Decision_prompt(agent1_label, agent2_label, agent3_label):
    # 生成推理分析模板，但是没有调用，因为我最后改成了直接投票的方式
    return f""" ### Instruction: You are a sarcastic judge arbitrator. By examining the labels given by each agent, 
    you ultimately determine the final label through voting and provide a label, where [0] represents "not sarcastic" 
    and [1] represents "sarcastic". 

    ### Agent Statements:
    Agent1 (Contextual): {agent1_label}
    Agent2 (Semantic): {agent2_label}
    Agent3 (Rhetorical): {agent3_label}

    ### Input:

    The final labels are determined through voting, Provide the label [0] or [1], and the answer can only be given with [0] or [1].




    ### Response:



    """


def Contextual_understanding_Agent(input_text):  # 语境理解Agent

    content = Contextual_understanding_prompt(input_text)  # 调用语境分析模板

    return content


def Semantic_analysis_Agent(input_text):  # 语义理解Agent
    content = Semantic_analysis_prompt(input_text)  # 调用语义理解模板

    return content


def Rhetorical_Analysis_Agent(input_text):  # 修辞分析Agent
    content = Rhetorical_Analysis_prompt(input_text)  # 调用修辞分析模板
    return content


def Reasoning_Decision_Agent(agent1_label, agent2_label, agent3_label):  # 推理与决策代理，直接投票的方式
    votes = [agent1_label, agent2_label, agent3_label]
    print(votes)
    vote0 = 0
    vote1 = 0
    for vote in votes:
        if vote == 0:
            vote0 += 1
        else:
            vote1 += 1
    if vote1 > vote0:
        label = 1
    else:
        label = 0

    print(vote0, vote1)
    return label


def eval_performance(y_true, y_pred, metric_path=None):  # 最后生成评价指标
    """评估模型性能指标"""
    # Precision
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


if __name__ == '__main__':
    # 数据集路径
    dataset_path = 'D:/PythonProject(LLM)/MyLLM/dataset/test_iacv1.csv'  # 数据集路径
    output_path = 'gpt-4o_io.csv'  # 输出路径
    metric_path = 'gpt-4o_iometric.json'  # 输出评价指标路径
    task_name = 'iacv1'  # 任务名字

    chunks = 32  # 将数据集划分成32块
    model = 'gpt-4o'  # 调用gpt-4o模型

    df = pd.read_csv(dataset_path, encoding_errors='ignore')  # 读取数据集
    df.dropna(inplace=True)  # 删除缺失值行

    chunk_size = int(np.ceil(len(df) / chunks))  # 计算每个分块的大小
    df_chunks = []  # 存储所有分块数据

    for chunk_num in range(chunks):
        all_texts = []  # 存储每一个分块的文本
        all_labels = []  # 储存每一个分块的标签
        chunk_file_path = output_path.replace('.csv', f'_{chunk_num + 1}.csv')  # 构建当前分块文件路径

        if os.path.exists(chunk_file_path):  # 检查分块文件是否存在
            df_chunk = pd.read_csv(chunk_file_path)  # 读取已存在的分块文件
            df_chunks.append(df_chunk)
            continue

        # 处理当前数据块
        df_chunk = df[chunk_num * chunk_size:min(len(df), (chunk_num + 1) * chunk_size)]  # 切分当前数据块

        Contextual_understanding_Agent_output_texts = []  # 存储语境分析代理输出结果
        Semantic_analysis_Agent_output_texts = []  # 存储语义分析代理输出结果
        Rhetorical_Analysis_Agent_output_texts = []  # 存储修辞分析代理输出结果
        Reasoning_Decision_Agent_output_texts = []  # 存储推理决策代理输出结果

        Contextual_understanding_Agent_labels = []  # 存储语境分析代理预测标签
        Semantic_analysis_Agent_labels = []  # 存储语义分析代理预测标签
        Rhetorical_Analysis_Agent_labels = []  # 存储修辞分析代理预测标签
        Reasoning_Decision_Agent_labels = []  # 存储推理决策代理输出标签

        for index, (_, row) in enumerate(tqdm(df_chunk.iterrows(), total=len(df_chunk),
                                              desc=f"Processing chunk {chunk_num + 1}/{chunks} for task")):
            input_text = row['Text']  # 获取数据集文本
            input_Label = row['Label']  # 获取数据集标签
            all_texts.append(input_text)
            all_labels.append(input_Label)

            content = Contextual_understanding_Agent(input_text)  # 语境分析代理
            print('语境分析代理正在处理中：')
            print("----------------")  # 打印分隔线
            result = Invoke_model(model, content)  # 调用模型，得到相应内容
            Contextual_understanding_Agent_output_texts.append(result)  # 存储语境分析代理给出的结果
            if '0' in result or '[0]' in result:  # 相应结果里有[0]或0，就给标签打0，否则打1
                pred = 0
            else:
                pred = 1
            Contextual_understanding_Agent_labels.append(pred)  # 存储语境分析代理给出的标签
            print("----------------")  # 打印分隔线

            content = Semantic_analysis_Agent(input_text)  # 语义分析代理
            print('语义分析代理正在处理中：')
            result = Invoke_model(model, content)  # 调用模型，得到相应内容
            Semantic_analysis_Agent_output_texts.append(result)  # 存储语义分析代理给出的标签
            if '0' in result or '[0]' in result:
                pred = 0
            else:
                pred = 1
            Semantic_analysis_Agent_labels.append(pred)  # 存储语义分析代理给出的标签
            print("----------------")  # 打印分隔线

            content = Rhetorical_Analysis_Agent(input_text)  # 修辞分析代理
            print('修辞分析代理正在处理中：')
            result = Invoke_model(model, content)  # 调用模型，得到相应内容
            Rhetorical_Analysis_Agent_output_texts.append(result)  # 存储修辞分析代理给出的结果
            if '0' in result or '[0]' in result:
                pred = 0
            else:
                pred = 1
            Rhetorical_Analysis_Agent_labels.append(pred)  # 存储修辞分析代理给出的标签
            print("----------------")  # 打印分隔线

            if Contextual_understanding_Agent_labels[-1] == Semantic_analysis_Agent_labels[-1] == \
                    Rhetorical_Analysis_Agent_labels[-1]:  # 如果三个代理给出的标签一致，直接给出答案
                print('各个代理得出的结论一致，可直接得出结论！')
                result = '各个代理得出的结论一致，可直接得出结论！'
                # 存储分析文本和标签
                Reasoning_Decision_Agent_output_texts.append(result)  #

                Reasoning_Decision_Agent_labels.append(Contextual_understanding_Agent_labels[-1])  # 随便一个代理的便签给最后的决策代理
            else:  # 如果有不一致的
                print('有代理结论不一致，进入推理与决策代理。')
                # 分别存储3个代理的标签
                agent1_label = Contextual_understanding_Agent_labels[-1]
                agent2_label = Semantic_analysis_Agent_labels[-1]
                agent3_label = Rhetorical_Analysis_Agent_labels[-1]
                # 调用决策代理函数
                result = Reasoning_Decision_Agent(agent1_label, agent2_label, agent3_label)

                Reasoning_Decision_Agent_output_texts.append(result)  # 存储决策代理结果

                Reasoning_Decision_Agent_labels.append(result) # 存储决策代理结果

        df_chunk = pd.DataFrame({
            "Text": all_texts, # Text列为原始文本
            "Label": all_labels, # Label列为原始标签
            "output": Reasoning_Decision_Agent_output_texts, # output列为决策代理的输出
            "pred": Reasoning_Decision_Agent_labels # pred列为决策代理最后给的标签
        })
        df_chunk.to_csv(chunk_file_path, index=False) # 保存该分块的文件
        df_chunks.append(df_chunk) # 将该分块的.csv文件加入到总.csv文件中

    df = pd.concat(df_chunks)  # 合并所有分块数据

    df.to_csv(output_path, index=0)  # 保存最终结果
    for i in range(chunks):
        chunk_file_path = output_path.replace('.csv', f'_{i}.csv')  # 构建分块文件路径模板
    eval_performance(df['Label'], df['pred'], metric_path)  # 执行性能评估
