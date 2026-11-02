"""Direct R/L feature concatenation on the three existing raw-image splits."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiment_anisometropia_frozen import evaluate, pair_features, save_json, sha256, validation_threshold
from report_anisometropia_modalities import add_table, metric_values
from report_raw_three_splits import ROOT, SPLITS, load_and_validate


def direct_features(eyes):
    eyes = np.asarray(eyes)
    if eyes.ndim != 3 or eyes.shape[1] != 2 or not np.isfinite(eyes).all():
        raise ValueError("Expected finite [patients, 2, embedding_dim] features in R/L order")
    return eyes.reshape(len(eyes), -1).astype(np.float64)


def fit_classifier(x, labels, split, seed):
    labels, split = np.asarray(labels), np.asarray(split)
    if len(x) != len(labels) or len(x) != len(split) or not np.isfinite(x).all():
        raise ValueError("Feature/label/split mismatch")
    if set(split) != set(SPLITS) or set(labels) != {0, 1}:
        raise ValueError("Expected three splits and binary known labels")
    train, val = split == "train", split == "val"
    scaler = StandardScaler().fit(x[train])
    z = scaler.transform(x)
    trials, best_auc, selected = [], -1., None
    for c in (.001, .01, .1, 1., 10.):
        head = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                  max_iter=3000, tol=1e-8, random_state=seed)
        head.fit(z[train], labels[train])
        if int(head.n_iter_.max()) >= head.max_iter:
            raise RuntimeError("Logistic regression did not converge")
        vm = evaluate(labels[val], head.predict_proba(z[val])[:, 1])
        trials.append({"C": c, "train": evaluate(labels[train], head.predict_proba(z[train])[:, 1]),
                       "val": vm, "iterations": head.n_iter_.tolist()})
        if vm["auc"] > best_auc:
            selected, best_auc, best_c = head, vm["auc"], c
    threshold = validation_threshold(labels[val], selected.predict_proba(z[val])[:, 1])
    probability = selected.predict_proba(z)[:, 1]
    np.testing.assert_allclose(probability, expit(z @ selected.coef_[0]+selected.intercept_[0]), atol=1e-12)
    return {"scaler": scaler, "head": selected, "probability": probability,
            "selected_C": best_c, "validation_threshold": threshold, "trials": trials,
            "metrics_at_0p5": {s: evaluate(labels[split == s], probability[split == s]) for s in SPLITS},
            "metrics_at_val_threshold": {s: evaluate(labels[split == s], probability[split == s], threshold)
                                         for s in SPLITS}}


def write_report(output, baseline, results):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.6)
    section.left_margin = section.right_margin = Cm(1.6)
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = doc.styles[name]
        style.font.name = "Microsoft YaHei"
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("双眼特征直接拼接消融实验", 0)
    doc.add_paragraph("原图 | seed=46、42、43 | 患者级是否屈光参差")
    doc.add_heading("数据与相同设置", 1)
    doc.add_paragraph(
        "沿用此前三次原图实验的患者名单及划分，不重新分组。601人按参差/不参差/未知标签分层划分"
        "400/100/101人，排除57名未知标签后，三组均为训练362人（122阳性、240阴性）、"
        "验证91人（31阳性、60阴性）、测试91人（31阳性、60阴性）。输入1088张原图，无年龄性别或屈光数据。")
    doc.add_paragraph(
        "直接复用共享冻结RETFound-Green已经提取的每眼384维特征[B,2,384]。"
        "原方案拼接双眼均值和绝对差[B,768]；新方案按右眼在前、左眼在后直接展开为[B,768]，"
        "不计算双眼均值、差值或绝对差。两方案均经过仅在训练集拟合的StandardScaler，"
        "再经768→1逻辑回归和sigmoid输出。可训练参数均为769，编码器保持冻结，无数据增强。"
        "保留的标准化是在训练患者间逐特征统计，不是对双眼取均值。")
    doc.add_paragraph(
        "C候选[0.001,0.01,0.1,1,10]，仅按验证AUC选择，并列取较小C；balanced类别权重、"
        "lbfgs、max_iter=3000、tol=1e-8与原实验一致。主阈值0.5，补充阈值仅按验证平衡准确率选择。"
        "各组C可因验证表现不同而不同，因此是相同调参规则的表示方式消融，不是强制固定C的对比。")
    doc.add_heading("测试结果对比（阈值0.5）", 1)
    rows = []
    for seed, result in results.items():
        for name, item in (("均值+绝对差", baseline[seed]["results"]), ("直接拼接", result)):
            m = item["metrics_at_0p5"]["test"]
            rows.append([f"{seed} {name}", item["selected_C"]]+[f"{m[k]:.4f}" for k in
                        ("auc", "acc", "balanced_acc", "recall", "f1")])
    add_table(doc, ["seed / 方案", "C", "AUC", "ACC", "BACC", "召回率", "F1"], rows)
    doc.add_paragraph(
        "直接拼接不再具有左右眼交换不变性，可能学习到眼别顺序相关特征。544人中88人的左右顺序"
        "仍沿用CSV、未重新图像核验，本轮为保持消融一致没有更改顺序，相关误差可能影响拼接方案。"
        "线性分类头不能等价实现绝对差这样的非线性关系。三次划分来自同一数据集且存在患者重叠，"
        "结果为探索性对照，不是独立外部验证，不能只依据测试结果选择最好的seed。")
    locations = []
    for seed, r in results.items():
        doc.add_page_break()
        doc.add_heading(f"seed={seed} 直接拼接结果", 1)
        doc.add_paragraph(f"所选C={r['selected_C']}，以下主结果阈值0.5。")
        add_table(doc, ["子集", "AUC", "ACC", "BACC", "精确率", "召回率", "F1"],
                  [metric_values(SPLITS[s], r["metrics_at_0p5"][s]) for s in SPLITS])
        doc.add_paragraph("混淆矩阵行是真实类别，列是预测类别；阳性表示屈光参差。")
        for split in SPLITS:
            doc.add_heading(SPLITS[split], 2)
            matrix = r["metrics_at_0p5"][split]["confusion_matrix"]
            locations.append((len(doc.tables), matrix))
            add_table(doc, ["真实 / 预测", "不参差", "参差"],
                      [["不参差", *matrix[0]], ["参差", *matrix[1]]])
        doc.add_heading("验证集阈值的补充结果", 2)
        doc.add_paragraph(f"阈值={r['validation_threshold']:.6f}；AUC不变，矩阵按[[TN,FP],[FN,TP]]排列。")
        add_table(doc, ["子集", "ACC", "召回率", "F1", "混淆矩阵"],
                  [[SPLITS[s], f"{m['acc']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", str(m["confusion_matrix"])]
                   for s in SPLITS for m in [r["metrics_at_val_threshold"][s]]])
        doc.add_paragraph(f"结果：{output / f'seed{seed}'}")
    path = output / "原图直接拼接三组消融实验报告.docx"
    doc.save(path)
    reopened = Document(path)
    if len(reopened.tables) != 16:
        raise ValueError("Report table count mismatch")
    for index, matrix in locations:
        shown = [[int(reopened.tables[index].cell(i, j).text) for j in (1, 2)] for i in (1, 2)]
        if shown != matrix:
            raise ValueError("Word confusion matrix mismatch")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "anisometropia_raw_direct_concat_seed46_42_43")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.set_num_threads(4)
    baseline = load_and_validate()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for seed, run in baseline.items():
        folder = Path(run["folder"])
        cache = folder / "retfound-green_raw_checkpoint/features_and_head.npz"
        with np.load(cache) as values:
            eyes = values["eyes"].copy()
        records = run["audit"]["records"]
        if any([e.get("verified_side", e.get("side")) for e in r["eyes"]] != ["R", "L"] for r in records):
            raise ValueError("Unexpected right/left feature order")
        labels = np.array([r["label"] for r in records])
        split = np.array([r["split"] for r in records])
        # Reproduce the original fitting procedure before changing representation.
        reproduced = fit_classifier(pair_features(eyes).astype(np.float64), labels, split, seed)
        old = run["results"]
        if reproduced["selected_C"] != old["selected_C"]:
            raise ValueError("Baseline head selection not reproduced")
        np.testing.assert_allclose(reproduced["validation_threshold"], old["validation_threshold"], atol=1e-12)
        np.testing.assert_allclose(reproduced["probability"], [p["probability"] for p in run["predictions"]], atol=1e-12)
        fitted = fit_classifier(direct_features(eyes), labels, split, seed)
        r = copy.deepcopy(old)
        for key in ("selected_C", "validation_threshold", "trials", "metrics_at_0p5", "metrics_at_val_threshold"):
            r[key] = fitted[key]
        r.update(pair_fusion="direct concatenation: right 384 then left 384; no inter-eye mean or difference",
                 pair_feature_order=["R", "L"], swap_invariant=False, frozen_feature_cache=str(cache),
                 frozen_feature_cache_sha256=sha256(cache), baseline_results=str(folder / "retfound-green_raw_checkpoint/results.json"))
        out = args.output / f"seed{seed}"
        out.mkdir()
        save_json(out / "data_audit.json", run["audit"])
        save_json(out / "protocol.json", {
            "seed": seed, "only_model_change": "replace inter-eye mean/absolute difference by ordered concatenation",
            "baseline_head_reproduced": True, "cache": str(cache), "cache_sha256": sha256(cache),
            "audit_sha256": sha256(folder / "data_audit.json"),
            "baseline_protocol_sha256": sha256(folder / "protocol.json"), "script_sha256": sha256(__file__),
            "test_used_for_selection": False, "C_grid": [.001, .01, .1, 1., 10.],
            "fresh_fit": "train-only scaler and linear head; existing frozen per-eye features reused",
            "eye_order_status_counts": dict(Counter(r["eye_order_status"] for r in records))})
        scaler, head, scores = (fitted[k] for k in ("scaler", "head", "probability"))
        np.savez_compressed(out / "features_and_head.npz", eyes=eyes, scaler_mean=scaler.mean_,
                            scaler_scale=scaler.scale_, coefficient=head.coef_, intercept=head.intercept_)
        save_json(out / "predictions.json", [{"record_id": item["record_id"], "patient_key": item["patient_key"],
            "split": item["split"], "label": item["label"], "probability": float(p), "prediction_0p5": int(p >= .5),
            "prediction_val_threshold": int(p >= r["validation_threshold"])} for item, p in zip(records, scores)])
        save_json(out / "results.json", r)
        results[seed] = r
        print(f"seed{seed}: " + json.dumps(r["metrics_at_0p5"]["test"]), flush=True)
    aggregate = {}
    for name, group in (("mean_abs", {s: b["results"] for s, b in baseline.items()}), ("direct_concat", results)):
        aggregate[name] = {metric: {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
            for metric in ("auc", "acc", "balanced_acc", "recall", "f1")
            for values in [[r["metrics_at_0p5"]["test"][metric] for r in group.values()]]}
    save_json(args.output / "summary.json", {"experiments": results, "test_aggregate_at_0p5": aggregate})
    print(write_report(args.output, baseline, results), flush=True)
    print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    main()
