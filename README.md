# 双眼原图弱视识别：LR 与 RBF-SVM

本仓库仅包含当前 **1100人队列**的两个原图实验：共享冻结的官方 RETFound-Green 编码器，分别连接逻辑回归和 RBF-SVM。两个分类头独立训练，不含表格输入、年龄性别或缺失掩码，也不使用图像增强。

权重各保留一个 seed：两个模型均按验证集 AUROC 从 seed 42、46、43 中选中 **seed42**，没有按测试成绩选模型。这里的结果不是三 seed 平均值，也不是早期1524人版本。测试成绩更高的其他 seed 未据此替换验证集选定模型。

## 文件

- `weights/retfound_green.pth`：共享冻结编码器，只保存一份。
- `weights/logistic_seed42.joblib`：训练集标准化器和独立LR分类头。
- `weights/rbf_svm_seed42.joblib`：训练集标准化器和独立RBF分类头。
- `core.py`、`train.py`、`predict.py`：独立训练、评价和离线推理代码，不依赖原项目绝对路径。
- `metrics/`、`selection.json`：汇总指标、混淆矩阵和验证集选 seed 的依据。
- `environment.json`、`requirements.txt`、`environment.yml`：实际依赖版本及环境要求。
- `verification.json`、`manifest.json`：导出复核及文件SHA256。
- `examples/manifest.example.csv`：虚构ID的输入格式示例，不包含患者资料。
- `original_source/`：原始实验源码、依赖模块及实际执行快照；`legacy_source/`保留更新前仓库的全部Python代码。
- `sample_dataset/`：经数据提供者确认公开授权的10名匿名患者双眼图像及二分类标签，共20张图像；`demo.py`可直接运行两个分类头。

除明确授权的10例匿名图像及标签外，公开内容不包括患者姓名、身份对应表、完整临床Excel、完整逐人预测、完整数据划分名单或特征缓存。SVM文件保留其推理所必需的支持向量，不附带患者身份信息。训练复跑仍需用户本地持有合法授权的数据，不能只凭本公开仓库重建私有研究队列。示例病例来自历史测试集，不是新的独立验证集。

## 本次发布的结果

队列为非弱视676人、单眼弱视424人；seed42划分为训练739人、验证188人、测试173人。下面是各保留权重的单seed测试结果，不是三seed均值。

| 分类器 | 验证AUROC | 测试AUROC | Accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LR seed42 | 0.7938 | 0.8019 | 71.10% | 68.75% | 72.48% | 0.6377 |
| RBF-SVM seed42 | 0.7939 | 0.8119 | 75.14% | 67.19% | 79.82% | 0.6667 |

默认阈值下混淆矩阵顺序为`[[TN,FP],[FN,TP]]`：LR为`[[79,30],[20,44]]`；RBF-SVM为`[[87,22],[21,43]]`。训练、验证、测试及补充阈值的完整指标在`metrics/`目录。

## 运行环境

已验证环境：Linux、Python 3.10.20、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、timm 0.9.12、scikit-learn 1.7.2。CPU可以推理和训练下游分类头，不要求GPU；原图批量编码使用GPU会更快。未声称已在所有操作系统和新建环境中验证。

```bash
conda create -n amblyopia-research python=3.10 -y
conda activate amblyopia-research
# CPU版本；CUDA版本可将cpu换成cu128，并确认本机驱动兼容
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify.py
python -m unittest discover -s tests -v
```

安装命令参考[PyTorch官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。保持scikit-learn/joblib版本一致，以减少序列化模型兼容性问题。`joblib`使用pickle机制，**仅加载可信来源的权重文件**。

## 离线推理

```bash
python demo.py
python predict.py --classifier logistic --right RIGHT.png --left LEFT.png --device cpu
python predict.py --classifier rbf_svm --right RIGHT.png --left LEFT.png --device cpu
```

输入为同一患者同次就诊的右眼、左眼原始彩色眼底图像。程序按右、左顺序组织输入。对已裁好的眼底图像可直接使用；未裁剪的相机整图应使用与训练一致的眼底框：

```bash
python predict.py --classifier logistic --right RIGHT.png --left LEFT.png \
  --right-crop 0 0 512 512 --left-crop 0 0 512 512
```

上面的坐标仅示范参数格式，必须换成实际图像的合法眼底框。不提供框时使用全图，程序不会自动定位眼底；不匹配的取景、相机标记或曝光可能影响预测。

预处理：EXIF转正、RGB转换、指定框裁剪、双线性等比例缩放、居中补黑至392×392，再按`(x/255-0.5)/0.5`归一化。

结构：`[B,2,3,392,392] → 共享冻结ViT → [B,2,384] → 双眼均值384＋绝对差384 → [B,768] → 训练集StandardScaler → LR或RBF-SVM → 每人一个分数`。编码器含12个Transformer块、384维token；图像patch为28×28，池化前去掉CLS和四个register token。

LR输出弱视概率，默认阈值0.5；RBF-SVM输出未校准决策分数，默认阈值0，**不是概率**。`--threshold validation_bacc`或`validation_sens90`可使用预先在验证集选定的补充阈值，不能据测试表现再选阈值。`validation_sens90`不保证新患者中的敏感度达到90%。输出是患者级二分类，不判断左右哪只眼弱视。

## 训练复跑

先在本地整理好临床纳排和双眼配对。研究队列排除了任一可用就诊记录中“特殊情况”非空或形觉剥夺明确阳性的患者，并按实际使用的就诊标签排除双眼弱视。非弱视不等同于无其他眼科问题。公开训练入口不会自行重新判定诊断或擅自筛选患者。

清单字段见`examples/manifest.example.csv`：每位患者仅一条选定就诊记录，`split`为`train/val/test`，`label`为`0/1`，右左图像路径相对清单目录或使用绝对路径。可选`right_crop/left_crop`为JSON坐标列表。同一患者不能跨集合。示例中的图片并不存在。

```bash
python train.py --manifest data/patients.csv --classifier logistic --seed 42 --output runs/logistic
python train.py --manifest data/patients.csv --classifier rbf_svm --seed 42 --output runs/rbf_svm
```

如需复用本地特征，`--cache PATH.npz`要求包含`eyes[N,2,384]`、`patient_id[N]`和`encoder_sha256`，按ID关联清单并检查编码器哈希；不得用不同裁剪或不同模型的缓存替代。本地生成的逐人预测和缓存属于私有研究输出，默认被`.gitignore`排除。

LR：balanced类别权重、lbfgs、max_iter=3000、tol=1e-8，C搜索`0.001/0.01/0.1/1/10`。RBF-SVM：balanced、probability=False、tol=1e-6、max_iter=100000，相同C网格，gamma搜索`scale/0.0001/0.001/0.01`。仅按验证AUROC选参，平局取预设网格首项；标准化仅拟合训练集；确定参数及阈值后才评估测试集。编码器不更新。

## 许可与限制

本发布依赖并分发RETFound-Green权重，保留其 **Custom Non-Commercial Research Licence**，见`LICENSE`和`NOTICE.md`。商业主体或涉及产业方的项目应遵照上游许可联系权利人，不应将本发布当成可无限制商业使用的软件或已获临床批准的器械。

本研究使用单中心、历史上已分析过的数据划分；测试结果仅为内部研究证据，不构成临床诊断或外部验证。尚需独立临床验证与数据授权。模型、环境与预处理任一变化都可能影响结果。
