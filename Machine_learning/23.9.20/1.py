import pandas as pd

# 创建一个示例 DataFrame 包含缺失值
df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})

# 移除包含缺失值的行
df_dropped = df.dropna()
print(df_dropped)
# Output:
#     A    B
# 0  1.0  5.0

# 在原始 DataFrame 上直接移除包含缺失值的行
df.dropna(inplace=True)
print(df)
# Output:
#     A    B
# 0  1.0  5.0