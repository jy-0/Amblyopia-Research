"""Validate and report three original-image experiments differing only by seed."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from scipy.special import expit

from create_anisometropia_601_split import make_split
from experiment_anisometropia_frozen import evaluate, save_json, sha256
from report_anisometropia_modalities import add_table, metric_values


ROOT = Path(__file__).resolve().parent / "outputs"
RUNS = {
    46: ROOT / "anisometropia_601_raw_frozen_seed46",
    42: ROOT / "anisometropia_601_raw_frozen_seed42_only_seed",
    43: ROOT / "anisometropia_601_raw_frozen_seed43_only_seed",
}
SPLITS = {"train": "训练集", "val": "验证集", "test": "测试集"}


def load_and_validate():
    runs = {}
    baseline_protocol = baseline_records = baseline_images = baseline_eyes = None
    for seed, folder in RUNS.items():
        audit = json.loads((folder / "data_audit.json").read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        result_dir = folder / "retfound-green_raw_checkpoint"
        results = json.loads((result_dir / "results.json").read_text())
        predictions = json.loads((result_dir / "predictions.json").read_text())
        manifest = Path(audit["split_manifest"])
        if sha256(manifest) != audit["split_manifest_sha256"]:
            raise ValueError("Split manifest changed")
        if sha256(audit["source"]) != audit["source_sha256"]:
            raise ValueError("Source CSV changed")
        records = json.loads(manifest.read_text())
        inventory = json.loads((manifest.parent / "summary.json").read_text())
        if results["seed"] != seed or protocol["seed"] != seed or inventory["seed"] != seed:
            raise ValueError("Seed mismatch")
        if [r["patient_key"] for r in records] != sorted(r["patient_key"] for r in records):
            raise ValueError("Patient ordering changed")
        if make_split(records, seed) != {i: r["split"] for i, r in enumerate(records)}:
            raise ValueError("Not the original label-only split algorithm")
        shared_protocol = {k: v for k, v in protocol.items() if k != "seed"}
        shared_records = [{k: v for k, v in r.items() if k != "split"} for r in records]
        images = [{k: r[k] for k in ("record_id", "patient_key", "label", "eyes")} for r in audit["records"]]
        with np.load(result_dir / "features_and_head.npz") as features:
            eyes = features["eyes"]
            if eyes.shape != (544, 2, 384):
                raise ValueError("Unexpected image feature dimensions")
            if baseline_protocol is None:
                baseline_protocol, baseline_records, baseline_images, baseline_eyes = (
                    shared_protocol, shared_records, images, eyes.copy())
            else:
                if (shared_protocol, shared_records, images) != (baseline_protocol, baseline_records, baseline_images):
                    raise ValueError("Experiments differ beyond seed and partition")
                np.testing.assert_array_equal(eyes, baseline_eyes)
            for prediction, record in zip(predictions, audit["records"]):
                if any(prediction[k] != record[k] for k in ("record_id", "patient_key", "split", "label")):
                    raise ValueError("Prediction identity/label mismatch")
            if len(predictions) != len(audit["records"]) or len({p["patient_key"] for p in predictions}) != 544:
                raise ValueError("Duplicate or missing predictions")
            x = np.concatenate([eyes.mean(1), np.abs(eyes[:, 0]-eyes[:, 1])], axis=1).astype(np.float64)
            train = np.array([p["split"] == "train" for p in predictions])
            np.testing.assert_allclose(features["scaler_mean"], x[train].mean(0), atol=1e-12)
            np.testing.assert_allclose(features["scaler_scale"], x[train].std(0), atol=1e-12)
            probability = expit(((x-features["scaler_mean"])/features["scaler_scale"]) @
                                features["coefficient"][0] + features["intercept"][0])
            np.testing.assert_allclose(probability, [p["probability"] for p in predictions], atol=1e-12)
        for key in ("metrics_at_0p5", "metrics_at_val_threshold"):
            for split in SPLITS:
                selected = [p for p in predictions if p["split"] == split]
                saved = results[key][split]
                recomputed = evaluate(np.array([p["label"] for p in selected]),
                                      np.array([p["probability"] for p in selected]), saved["threshold"])
                if saved["confusion_matrix"] != recomputed["confusion_matrix"]:
                    raise ValueError("Confusion matrix mismatch")
                for metric in saved.keys()-{"confusion_matrix"}:
                    np.testing.assert_allclose(saved[metric], recomputed[metric], atol=1e-12)
        runs[seed] = {"folder": str(folder), "results": results, "inventory": inventory,
                      "audit": audit, "predictions": predictions}
    return runs


def main():
    runs = load_and_validate()
    overlaps = []
    for first, second in combinations(runs, 2):
        item = {"seeds": [first, second]}
        for split in SPLITS:
            groups = [{p["patient_key"] for p in runs[s]["predictions"] if p["split"] == split}
                      for s in (first, second)]
            item[split] = len(groups[0] & groups[1])
        overlaps.append(item)
    metrics = ("auc", "acc", "balanced_acc", "precision", "recall", "f1")
    aggregate = {}
    for metric in metrics:
        values = [r["results"]["metrics_at_0p5"]["test"][metric] for r in runs.values()]
        aggregate[metric] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}

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
    doc.add_heading("三次原图实验汇总", 0)
    doc.add_paragraph("seed=46、42、43 | 相同标签分层算法 | 患者级是否屈光参差")
    doc.add_heading("1. 数据集与划分", 1)
    doc.add_paragraph(
        "数据来源：/root/the_dataset 1.csv。601名患者具有可确认配对的双眼原图；其中544人本次检查的"
        "屈光参差标签明确（184人参差、360人不参差），共1088张图像进入实验。57人标签未知，"
        "不用于训练、验证选参或测试指标，不将未知当作阴性，也不借用其他就诊记录的标签。")
    doc.add_paragraph(
        "三次均将同一601人名单按姓名键排序，按参差/不参差/未知三层进行患者级分层随机划分。"
        "先取400人训练，其余201人再分为100人验证、101人测试；两步分别使用该次相同seed。"
        "随后排除各子集中的未知标签。每人双眼始终位于同一个子集，不额外按黑白图可用性分层。"
        "本报告不包含此前额外按黑白图可用性分层的seed42实验。")
    rows = []
    for seed, run in runs.items():
        for split in SPLITS:
            c = run["inventory"]["by_split"][split]
            rows.append([seed, SPLITS[split], c["patients"], c["unknown"], c["supervised_raw"],
                         c["positive"], c["negative"]])
    add_table(doc, ["seed", "子集", "划分人数", "未知排除", "实验人数", "参差", "不参差"], rows)
    doc.add_paragraph(
        "总划分比例400:100:101，约4:1:1；监督学习的训练/验证/测试均为362/91/91人。"
        "人数和类别比例相同不代表患者名单相同。各次逐人名单见对应划分目录的patient_split.csv。")
    doc.add_heading("相同子集的患者重合人数", 2)
    add_table(doc, ["两次seed", "训练集重合", "验证集重合", "测试集重合"],
              [[f"{v['seeds'][0]} / {v['seeds'][1]}", v["train"], v["val"], v["test"]] for v in overlaps])
    doc.add_paragraph(
        "本轮核验：三次协议除seed外一致；入组记录、标签、图像路径、裁剪框及原图哈希一致；"
        "544人的冻结编码器输出特征逐元素一致。标准化器和分类头仅用各次训练集重新拟合。")

    doc.add_page_break()
    doc.add_heading("2. 模型与实验结果", 1)
    doc.add_paragraph(
        "输入[B,2,3,392,392]：两张原始RGB眼底照经相同视野裁剪、双线性等比例缩放及黑边填充，"
        "各通道均值/标准差均为0.5。共享冻结RETFound-Green ViT-S/14 → 每眼384维[B,2,384]"
        " → 双眼均值与绝对差拼接[B,768] → 训练集StandardScaler → L2逻辑回归 → sigmoid[B,1]。"
        "编码器21,832,704个参数冻结，分类头769个参数可训练。无年龄性别或屈光数据输入，无数据增强。")
    doc.add_paragraph(
        "类别权重采用训练集balanced权重；solver=lbfgs、max_iter=3000、tol=1e-8。"
        "C仅从[0.001,0.01,0.1,1,10]中按验证AUC选择，并列选更小C。"
        "主结果固定阈值0.5；补充阈值仅按验证集平衡准确率选择，测试集不参与调参。")
    add_table(doc, ["seed", "所选C", "验证集所选阈值"],
              [[seed, r["results"]["selected_C"], f"{r['results']['validation_threshold']:.6f}"]
               for seed, r in runs.items()])
    doc.add_heading("固定阈值0.5的完整指标", 2)
    add_table(doc, ["seed / 子集", "AUC", "ACC", "BACC", "精确率", "召回率", "F1"],
              [metric_values(f"{seed} / {SPLITS[split]}", r["results"]["metrics_at_0p5"][split])
               for seed, r in runs.items() for split in SPLITS])
    doc.add_heading("三次测试结果的描述性汇总", 2)
    add_table(doc, ["统计量", "AUC", "ACC", "BACC", "精确率", "召回率", "F1"],
              [["均值 ± 标准差"]+[f"{aggregate[k]['mean']:.4f} ± {aggregate[k]['sample_std']:.4f}" for k in metrics]])
    doc.add_paragraph(
        "标准差为三次结果的样本标准差（ddof=1），只描述本次划分波动，不是置信区间或显著性检验。"
        "同一患者可能在不同实验中重复进入测试集，三次并非三个独立外部队列；不合并为273名独立患者，"
        "也不挑测试表现最好的seed作为最终性能。AUC衡量排序，ACC、F1和召回率还依赖阈值，需同时查看。")

    locations = []
    for number, (seed, run) in enumerate(runs.items(), start=3):
        r = run["results"]
        doc.add_page_break()
        doc.add_heading(f"{number}. seed={seed} 混淆矩阵", 1)
        doc.add_paragraph("以下混淆矩阵均使用阈值0.5；行是真实类别，列是预测类别。阳性为屈光参差。")
        for split in SPLITS:
            matrix = r["metrics_at_0p5"][split]["confusion_matrix"]
            doc.add_heading(SPLITS[split], 2)
            locations.append((len(doc.tables), matrix))
            add_table(doc, ["真实 / 预测", "无屈光参差", "有屈光参差"],
                      [["无屈光参差", *matrix[0]], ["有屈光参差", *matrix[1]]])
        doc.add_heading("验证集阈值下的补充结果", 2)
        doc.add_paragraph(
            f"本组验证集选择阈值={r['validation_threshold']:.6f}。AUC不随阈值改变。"
            "下表混淆矩阵按[[TN,FP],[FN,TP]]排列，不与上方0.5阈值矩阵混用。")
        add_table(doc, ["子集", "ACC", "召回率", "F1", "[[TN,FP],[FN,TP]]"],
                  [[SPLITS[s], f"{m['acc']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", str(m["confusion_matrix"])]
                   for s in SPLITS for m in [r["metrics_at_val_threshold"][s]]])
        doc.add_paragraph(f"结果目录：{run['folder']}")
        doc.add_paragraph(f"患者划分：{Path(run['audit']['split_manifest']).parent / 'patient_split.csv'}")

    output = ROOT / "reports/原图三次划分实验汇总_seed46_42_43.docx"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    reopened = Document(output)
    if len(reopened.tables) != 17:
        raise ValueError("Report table count mismatch")
    for index, matrix in locations:
        shown = [[int(reopened.tables[index].cell(i, j).text) for j in (1, 2)] for i in (1, 2)]
        if shown != matrix:
            raise ValueError("Word confusion matrix mismatch")
    summary = {"seeds": list(runs), "test_aggregate_at_0p5": aggregate, "split_overlaps": overlaps,
               "validation": "protocol differs only by seed; identical cohort and embeddings; train-only scaler; all 18 metric sets recomputed; 9 Word matrices checked",
               "experiments": {str(s): r["results"] for s, r in runs.items()},
               "report": str(output), "report_sha256": sha256(output)}
    save_json(output.parent / "raw_three_splits_seed46_42_43_summary.json", summary)
    print(output, flush=True)
    print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
