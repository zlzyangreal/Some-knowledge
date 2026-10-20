## Pandas 的工作原理
Pandas 开源库内提供了 DataFrame。DataFrame 是二维数组式数据表，其中每列包含一个变量的值，每行包含一组每列的值。DataFrame 中存储的数据可以是数字、系数，也可以是字符类型。也可以将 Pandas DataFrame 视作系列对象的词典或集合。
这表明 Pandas 主要用于 DataFrame 形式的机器学习。

![DataFrame。](https://www.nvidia.cn/content/dam/en-zz/Solutions/glossary/data-science/pandas/img-1.png "DataFrame。")

Pandas 支持导入和导出不同格式的表格数据，如 CSV 或 JSON 文件。

![导入和导出表格数据。](https://www.nvidia.cn/content/dam/en-zz/Solutions/glossary/data-science/pandas/img-2.png "导入和导出表格数据。")

此外，Pandas 还支持各种数据操作运算和数据清理功能，包括选择子集、创建衍生列、排序、连接、填充、替换、汇总统计数据和绘图。

![](https://www.nvidia.cn/content/dam/en-zz/Solutions/glossary/data-science/pandas/img-3.png "数据操作：排序。")

![](https://www.nvidia.cn/content/dam/en-zz/Solutions/glossary/data-science/pandas/img-4.png "数据操作：连接。")

![](https://www.nvidia.cn/content/dam/en-zz/Solutions/glossary/data-science/pandas/img-5.png "数据操作：汇总统计数据和绘图。")

Python 软件包索引（Python 编程语言软件库）的编制者表示，Pandas 非常适合处理多种数据，包括：

- 包含异构类型列的表格数据，如 SQL 表或 Excel 电子表格
- 有序和无序（可能并非固定频率）时间序列数据
- 具有行和列标签的任意矩阵数据（同构类型或异构类型）

其他任何形式的可观察/统计数据集。实际上，数据完全无需标记即可放入 pandas 数据结构中。

## Pandas 的优势

此外，Python 软件包索引的编制者还表示，Pandas 为数据科学家和开发者提供了几个关键优势，包括：

- 轻松处理浮点和非浮点数据中的缺失数据（表示为 NaN）
- 大小易变性：可以从 DataFrame 和更高维度的对象中插入和删除列
- 自动和显式数据对齐：可以将对象显式对齐到一组标签；或者用户只需忽略标签，让序列、DataFrame 等在计算中自动调整数据
- 强大、灵活的分组功能，对数据集执行分割-应用-组合操作，进行数据聚合和转换
- 可轻松将其他 Python 和 Numpy 数据结构中参差不齐、索引不同的数据转换为 DataFrame 对象
- 大型数据集基于标签的智能切片、精美索引和子集构建
- 直观的数据集合并与连接
- 灵活的数据集重塑和旋转
- 坐标轴的分层标记（每个记号可能具有多个标签）
- 强大的 I/O 工具，用于加载平面文件（CSV 和分隔文件）、Excel 文件和数据库中的数据，以及保存/加载超快速 HDF5 格式的数据
- 特定于时间序列的功能：日期范围生成和频率转换、窗口统计数据迁移、日期调整和延迟

Pandas 库的其他优势还包括：缺失数据的数据对齐和集成处理；数据集合并与连接；数据集重塑和旋转；分层轴索引（以便在低维数据结构中处理高维数据）以及基于标签的切片。