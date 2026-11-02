"""Word report from the recorded seed-42 paired-image experiments."""
import argparse
import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


NAMES = {"raw": "双眼原图", "binary": "双眼黑白反光图"}
SPLITS = {"train": "训练集", "val": "验证集", "test": "测试集"}


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
    repeat = OxmlElement("w:tblHeader")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for values in rows:
        for cell, value in zip(table.add_row().cells, values):
            cell.text = str(value)
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(9)
    return table


def metric_values(name, metrics):
    return [name]+[f"{metrics[k]:.4f}" for k in ("auc", "acc", "balanced_acc", "precision", "recall", "f1")]


def build_report(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    experiments, common = summary["experiments"], summary["common_test"]
    protocol = summary["protocol"]
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.6)
    section.left_margin = section.right_margin = Cm(1.6)
    for name in ("Normal", "Title", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.name = "Microsoft YaHei"
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    doc.styles["Normal"].font.size = Pt(10)
    doc.styles["Normal"].paragraph_format.space_after = Pt(5)
    doc.add_heading("原图与黑白反光图实验报告", 0)
    doc.add_paragraph(f"seed={protocol['seed']} | 患者级是否屈光参差 | 冻结RETFound-Green | 两组实验")
    doc.add_heading("1. 数据与划分", 1)
    doc.add_paragraph(
        "来源为 /root/the_dataset 1.csv。只使用本次检查标签明确、双眼图片可唯一匹配且文件核验通过的患者；"
        "未知标签不当作阴性，也不借用另一就诊记录的标签。原图共601人，其中544人可监督学习；"
        "已有双眼整幅黑白反光图482人，其中456人可监督学习。这是当前已确认同次检查配对队列中的最大可用人数。")
    rows = []
    for kind in NAMES:
        inv = protocol[kind+"_inventory_counts"]
        for split in SPLITS:
            c = experiments[kind]["split_counts"][split]
            rows.append([NAMES[kind], SPLITS[split], inv[split]["patients"], inv[split]["unknown"],
                         c["patients"], c["positive"], c["negative"]])
    add_table(doc, ["输入", "划分", "含未知人数", "未知排除", "实际人数", "参差", "不参差"], rows)
    doc.add_paragraph(
        "重新划分，不沿用seed=46的子集。目标比例为400:100:101（约4:1:1）；按是否有双眼黑白图分层，"
        "每层再按阳性、阴性、未知标签进行seed=42分层抽样。黑白图含未知标签人数为321/80/81，"
        "监督学习人数为304/76/76；原图监督学习人数为362/91/91。整数取整导致比例有微小差别。"
        "同一患者的两眼始终在一起，且共有患者在两个实验中属于同一子集。")
    doc.add_paragraph(
        "原图阳性184/544（33.82%），黑白图阳性96/456（21.05%），不是类别平衡数据集。"
        "原图多出的88名已标注患者全部为阳性，缺图具有选择性，因此完整测试集结果并非严格单因素对照。"
        "另有119人缺双眼黑白图，其中31人标签未知；本轮没有生成或补入新分割图。")
    doc.add_heading("2. 相同模型与训练流程", 1)
    doc.add_paragraph(
        "每人两眼 [B,2,3,392,392] → 展平 [2B,3,392,392] → 共享冻结的RETFound-Green ViT-S/14"
        " → 每眼384维 [B,2,384] → 拼接双眼均值384维和绝对差384维 [B,768]"
        " → 训练集StandardScaler [B,768] → L2逻辑回归 [B,1] → sigmoid屈光参差分数。"
        "输出为每人一个分数，不是每眼一个分数，不判断哪只眼弱视。")
    doc.add_paragraph(
        "编码器21,832,704个参数全部冻结，分类头769个可训练参数。两组使用同一预训练权重，"
        "分别重新拟合标准化参数和分类头，不复用旧模型分类头或旧特征缓存。无年龄、性别、屈光参数输入，"
        "无缺失掩码输入，无数据增强。")
    doc.add_paragraph(
        "两组使用相同原图眼底视野裁剪框、等比例缩放并居中补黑边。原图为RGB双线性缩放；"
        "黑白图检查像素属于{0,1,255}，按>0二值化，最近邻缩放后复制为3通道。"
        "均值和标准差每通道均为0.5。黑白图实验不读取原图颜色作为模型特征。")
    doc.add_paragraph(
        "仅用训练集拟合标准化器和类别平衡权重；逻辑回归solver=lbfgs，max_iter=3000，tol=1e-8。"
        "C从[0.001,0.01,0.1,1,10]中按验证AUC选择，并列选更小C。"
        "主结果阈值固定为0.5；补充阈值仅按验证集平衡准确率选择。测试集不参与调参。"
        "冻结特征线性探针不进行ViT的epoch微调。")
    headers = ["数据集", "AUC", "ACC", "BACC", "精确率", "召回率", "F1"]
    for number, kind in enumerate(NAMES, start=3):
        doc.add_page_break()
        r = experiments[kind]
        doc.add_heading(f"{number}. {NAMES[kind]}结果", 1)
        doc.add_paragraph(f"选择C={r['selected_C']}；以下主结果统一阈值0.5。阳性表示屈光参差。")
        add_table(doc, headers, [metric_values(SPLITS[s], r["metrics_at_0p5"][s]) for s in SPLITS])
        doc.add_heading("训练、验证、测试混淆矩阵", 2)
        doc.add_paragraph("行是真实类别，列是预测类别；左上TN、右上FP、左下FN、右下TP。")
        for split in SPLITS:
            matrix = r["metrics_at_0p5"][split]["confusion_matrix"]
            doc.add_heading(SPLITS[split], 3)
            add_table(doc, ["真实 / 预测", "无屈光参差", "有屈光参差"],
                      [["无屈光参差", *matrix[0]], ["有屈光参差", *matrix[1]]])
        doc.add_heading("仅用验证集选择的阈值", 2)
        doc.add_paragraph(f"阈值={r['validation_threshold']:.6f}；AUC与阈值0.5时相同。")
        add_table(doc, ["数据集", "ACC", "召回率", "F1", "[[TN,FP],[FN,TP]]"],
                  [[SPLITS[s], f"{m['acc']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", str(m["confusion_matrix"])]
                   for s in SPLITS for m in [r["metrics_at_val_threshold"][s]]])
    doc.add_page_break()
    doc.add_heading("5. 测试结果对比", 1)
    doc.add_paragraph("完整测试集：各自最大样本量训练后的表现，阈值均为0.5。")
    add_table(doc, ["输入 / 测试人数", "AUC", "ACC", "BACC", "精确率", "召回率", "F1"],
              [metric_values(f"{NAMES[k]} / {r['split_counts']['test']['patients']}", r["metrics_at_0p5"]["test"])
               for k, r in experiments.items()])
    doc.add_heading("共同测试患者", 2)
    doc.add_paragraph(
        f"同一批{common['patients']}人（参差{common['positive']}人、不参差{common['negative']}人），"
        "均未进入任一模型的训练或验证集。只重新汇总已保存预测，不重新训练、不选择新阈值。")
    add_table(doc, headers, [metric_values(NAMES[k], common["metrics_at_0p5"][k]) for k in NAMES])
    add_table(doc, ["输入", "阈值0.5混淆矩阵", "本组验证阈值", "验证阈值混淆矩阵"],
              [[NAMES[k], str(common["metrics_at_0p5"][k]["confusion_matrix"]),
                f"{experiments[k]['validation_threshold']:.6f}",
                str(common["metrics_at_val_threshold"][k]["confusion_matrix"])] for k in NAMES])
    raw_m, binary_m = (experiments[k]["metrics_at_0p5"]["test"] for k in ("raw", "binary"))
    doc.add_heading("结论与边界", 2)
    doc.add_paragraph(
        f"完整测试集：原图AUC={raw_m['auc']:.4f}、F1={raw_m['f1']:.4f}；"
        f"黑白图AUC={binary_m['auc']:.4f}、F1={binary_m['f1']:.4f}。"
        f"共同测试集：原图AUC={common['metrics_at_0p5']['raw']['auc']:.4f}，"
        f"黑白图AUC={common['metrics_at_0p5']['binary']['auc']:.4f}。"
        "共同测试对比排除了测试人群差异，但两组训练与验证人数、阳性比例仍不同，不能把差异只归因于输入类型。")
    for kind, r in experiments.items():
        m = r["metrics_at_0p5"]
        negative = r["split_counts"]["test"]["negative"]
        count = r["split_counts"]["test"]["patients"]
        doc.add_paragraph(
            f"{NAMES[kind]}：训练/验证/测试AUC分别为{m['train']['auc']:.4f}/"
            f"{m['val']['auc']:.4f}/{m['test']['auc']:.4f}。全部预测为阴性的测试准确率基线为"
            f"{negative/count:.2%}，应结合召回率、F1及BACC理解准确率。")
    doc.add_paragraph(
        "仅单seed探索性实验，没有独立外部验证，不证明稳定优劣。新划分患者与历史实验重叠，"
        "本轮分类头没有训练/测试泄漏，但该数据集已被反复用于实验，不是全新的外部测试集。"
        "现有黑白图的分割器训练来源未确认；如分割器曾使用测试图人工训练或调参，仍可能引入额外偏倚。"
        "原图新增已标注患者中88人的左右顺序沿用CSV，未重新人工核验；本模型均值与绝对差融合对双眼交换不敏感。"
        "类别平衡训练后的sigmoid分数未做风险概率校准。")
    doc.add_heading("6. 文件与复现", 1)
    doc.add_paragraph(f"实验目录：{root}")
    doc.add_paragraph(
        "patient_split.csv保存601人划分及两组纳入状态；raw_data_audit.json、binary_data_audit.json保存实际输入清单；"
        "各组results.json保存完整指标，predictions.json保存逐人预测，features_and_head.npz保存特征及分类头；"
        "common_test_results.json保存共同测试比较，protocol.json保存种子、版本和校验值。")
    doc.add_paragraph(
        "conda activate hospital\ncd /root/hospital\n"
        "python model_code/experiment_anisometropia_modalities.py --seed 42 "
        "--output model_code/outputs/anisometropia_raw_binary_max_seed42_repeat\n"
        "重复运行需指定新的输出目录，程序拒绝覆盖已有结果。")
    path = root / f"原图与黑白反光图实验报告_seed{protocol['seed']}.docx"
    doc.save(path)
    reopened = Document(path)
    if len(reopened.tables) != 14:
        raise AssertionError(f"Unexpected report table count: {len(reopened.tables)}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(build_report(parser.parse_args().directory))
