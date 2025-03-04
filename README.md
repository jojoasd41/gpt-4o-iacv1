是用投票的方式，最后的结果为：
{
    "Precision": 0.75,
    "Recall": 0.8363636363636363,
    "Accuracy": 0.771875,
    "F1": 0.7908309455587392,
    "Micro-F1": 0.771875,
    "Macro-F1": 0.7699859195147648,
    "Weighted-F1": 0.7706373265786389
}
我这个每一块的结果都是保存了的，如果跑着跑着，中间断了，不用从头跑。只需要改下代码里这个位置，就可以从断的位置开始跑了：
    for chunk_num in range('从第几块开始断的，就写几',chunks):
        all_texts = []  # 存储每一个分块的文本
        all_labels = []  # 储存每一个分块的标签
        chunk_file_path = output_path.replace('.csv', f'_{chunk_num + 1}.csv')  # 构建当前分块文件路径

        if os.path.exists(chunk_file_path):  # 检查分块文件是否存在
            df_chunk = pd.read_csv(chunk_file_path)  # 读取已存在的分块文件
            df_chunks.append(df_chunk)
            continue
如果中途断了，最后的评估结果就用代码Concat，把中途产生的所有32块.csv文件合一起进行评估；如果中途没断，那当我没说。
