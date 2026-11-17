# ITC 引用情感计算与分析代码（PyCharm版）

## 你需要准备的文件

只需要一个文件：

```text
隐形三角引用数据.xlsx
```

放置结构：

```text
工作目录/
├── 隐形三角引用数据.xlsx
└── itc_sentiment_pipeline/
```

训练语料由代码从 Athar Citation Sentiment Corpus 官方地址自动下载、清洗和转换，不需要手工准备。

## 第一次使用

建议使用 Python 3.10 或 3.11。在 PyCharm Terminal 中进入 `itc_sentiment_pipeline`，安装依赖：

```bash
pip install -r requirements.txt
```

检查ITC数据：

```bash
python smoke_test.py
```

出现 `Input smoke test PASS` 后继续。

## 第一阶段：自动准备语料、训练并预测

运行：

```bash
python run_pipeline.py all
```

代码自动完成：

1. 下载并清洗 Athar 2011 引用情感语料；
2. 训练、验证和测试 BiomedBERT 情感模型；
3. 读取6,165条ITC记录；
4. 规范化并去重引用句；
5. 计算消极、中性、积极标签、概率、置信度和极性分；
6. 生成人工复核表和独立盲评表。

第一阶段结束后会暂停，因为人工标签必须由研究者判断。

## 唯一需要人工完成的环节

### 文件1：标签复核

打开：

```text
outputs/02_review_to_complete.xlsx
```

填写 `human_label`：

```text
negative
neutral
positive
uncertain
```

保存为：

```text
outputs/02_review_completed.xlsx
```

### 文件2：独立盲评

打开：

```text
outputs/03_validation_to_complete.xlsx
```

填写 `human_label`，保存为：

```text
outputs/03_validation_completed.xlsx
```

盲评完成前不要打开 `outputs/03_validation_key.csv`。

## 第二阶段：自动统计和画图

人工填写完成后运行：

```bash
python run_pipeline.py analyze
python run_pipeline.py figures
```

代码自动计算：

- 情感分布；
- 两条引用句的情感一致率；
- Cohen's Kappa；
- Cramér's V；
- 连续极性分的 Spearman 相关；
- 文献三元组聚类 Bootstrap 95% CI；
- 引用位置和引用功能的情感分布；
- 0.50、0.60、0.70、0.80置信度敏感性分析；
- 模型人工验证指标；
- 论文图表。

核心输出：

```text
outputs/04_sentence_final.csv
outputs/04_pair_level_final.xlsx
outputs/04_validation_metrics.json
outputs/05_statistical_results.xlsx
outputs/figures/
```

## 完整命令顺序

```bash
pip install -r requirements.txt
python smoke_test.py
python run_pipeline.py all

# 填写并保存两个Excel

python run_pipeline.py analyze
python run_pipeline.py figures
```

## 分析边界

当前数据全部为ITC记录，可以分析ITC内部的情感分布、情感一致性、引用位置和引用功能差异。没有普通引用对照组时，不能表述为“ITC比普通引用更积极或更消极”。
