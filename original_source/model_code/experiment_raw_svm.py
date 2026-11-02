"""Validation-selected SVM probes on the unchanged frozen raw-fundus features."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import torch

from experiment_anisometropia_frozen import evaluate, pair_features, save_json, sha256
from experiment_reproduced_raw_comparison import SEEDS, TASKS, SPLITS, METRICS, load_comparisons

ROOT = Path(__file__).resolve().parent / 'outputs'
C_GRID = (.001, .01, .1, 1., 10.)
GAMMA_GRID = ('scale', .0001, .001, .01)
MODELS = ('logistic', 'linear_svm', 'rbf_svm', 'selected_svm')


def margin_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Finite scores and both validation classes required')
    unique = np.unique(scores)
    candidates = np.unique(np.r_[0., np.nextafter(unique[0], -np.inf),
                                np.nextafter(unique[-1], np.inf), (unique[:-1]+unique[1:])/2])
    return float(max(candidates, key=lambda t: (
        balanced_accuracy_score(labels, scores >= t), -abs(t), -t)))


def fit_svm(x, labels, split, kernel, seed, linear_solver='svc'):
    x, labels, split = np.asarray(x, dtype=np.float64), np.asarray(labels), np.asarray(split)
    if kernel not in ('linear', 'rbf'):
        raise ValueError('Expected linear or rbf kernel')
    if linear_solver not in ('svc', 'linear_svc'):
        raise ValueError('Expected svc or linear_svc solver')
    if x.ndim != 2 or len(x) != len(labels) or len(x) != len(split) or not np.isfinite(x).all():
        raise ValueError('Feature/label/split mismatch or nonfinite features')
    if set(split) != set(SPLITS) or any(set(labels[split == s]) != {0, 1} for s in SPLITS):
        raise ValueError('Three splits with both binary classes required')
    train, val = split == 'train', split == 'val'
    scaler = StandardScaler().fit(x[train])
    z_train, z_val = scaler.transform(x[train]), scaler.transform(x[val])
    best, best_auc, trials = None, -np.inf, []
    for c in C_GRID:
        for gamma in (GAMMA_GRID if kernel == 'rbf' else ('scale',)):
            primal = kernel == 'linear' and linear_solver == 'linear_svc'
            model = (LinearSVC(C=c, class_weight='balanced', dual=False, loss='squared_hinge',
                               tol=1e-6, max_iter=100000, random_state=seed) if primal else
                     SVC(C=c, kernel=kernel, gamma=gamma, class_weight='balanced',
                         probability=False, tol=1e-6, max_iter=100000, cache_size=512, random_state=seed))
            model.fit(z_train, labels[train])
            if (primal and model.n_iter_ >= model.max_iter) or (not primal and model.fit_status_ != 0):
                raise RuntimeError('SVM did not converge')
            assert model.classes_.tolist() == [0, 1]
            val_score = model.decision_function(z_val)
            auc = float(roc_auc_score(labels[val], val_score))
            params = {'kernel': kernel, 'C': c, 'gamma': gamma if kernel == 'rbf' else None}
            trials.append({'parameters': params, 'train': evaluate(labels[train], model.decision_function(z_train), 0.),
                           'val': evaluate(labels[val], val_score, 0.)})
            if auc > best_auc:
                best, best_auc, selected_params = model, auc, params
    threshold = margin_threshold(labels[val], best.decision_function(z_val))
    # Evaluate test scores only after all parameters and thresholds are fixed.
    scores = best.decision_function(scaler.transform(x))
    result = {'classifier': kernel + '_svm', 'selected_parameters': selected_params,
              'selected_validation_auc': best_auc, 'default_threshold': 0.,
              'validation_threshold': threshold, 'score_type': 'uncalibrated signed decision margin, not probability',
              'class_weight': 'balanced', 'probability': False, 'trials': trials,
              'solver': 'liblinear_primal' if kernel == 'linear' and linear_solver == 'linear_svc' else 'libsvm',
              'loss': 'squared_hinge' if kernel == 'linear' and linear_solver == 'linear_svc' else 'hinge',
              'support_vectors_per_class': best.n_support_.tolist() if hasattr(best, 'n_support_') else None,
              'support_vector_count': int(best.n_support_.sum()) if hasattr(best, 'n_support_') else None,
              'effective_gamma': float(best._gamma) if kernel == 'rbf' else None,
              'metrics_at_default_threshold': {s: evaluate(labels[split == s], scores[split == s], 0.) for s in SPLITS},
              'metrics_at_val_threshold': {s: evaluate(labels[split == s], scores[split == s], threshold) for s in SPLITS}}
    return {'scaler': scaler, 'model': best, 'scores': scores, 'result': result}


def wrapped_logistic(result):
    return {'classifier': 'logistic', 'selected_parameters': {'C': result['selected_C']},
            'default_threshold': .5, 'validation_threshold': result['validation_threshold'],
            'score_type': 'predicted positive probability',
            'metrics_at_default_threshold': result['metrics_at_0p5'],
            'metrics_at_val_threshold': result['metrics_at_val_threshold']}


def verify_predictions(records, scores, result):
    for metrics_key, threshold in (('metrics_at_default_threshold', result['default_threshold']),
                                   ('metrics_at_val_threshold', result['validation_threshold'])):
        for split in SPLITS:
            indices = np.array([r['split'] == split for r in records])
            labels = np.array([r['label'] for r in records])[indices]
            measured = evaluate(labels, np.asarray(scores)[indices], threshold)
            assert measured['confusion_matrix'] == result[metrics_key][split]['confusion_matrix']
            for key in (*METRICS, 'threshold'):
                np.testing.assert_allclose(measured[key], result[metrics_key][split][key], rtol=0, atol=1e-12)


def write_report(output, summary):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    from report_anisometropia_modalities import add_table
    names = {'logistic': '原逻辑回归', 'linear_svm': '线性SVM', 'rbf_svm': 'RBF-SVM', 'selected_svm': '验证集选定SVM'}
    tasks = {'amblyopia': '弱视', 'anisometropia': '屈光参差'}
    protocol = summary['protocol']
    patients = protocol.get('patients', 544)
    images = protocol.get('raw_images', patients * 2)
    counts = protocol.get('split_counts', {'train': 362, 'val': 91, 'test': 91})
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(1.3)
    section.top_margin = section.bottom_margin = Cm(1.3)
    for name in ('Normal', 'Title', 'Heading 1', 'Heading 2'):
        style = doc.styles[name]
        style.font.name = 'Microsoft YaHei'
        style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    doc.styles['Normal'].font.size = Pt(9)
    def table(headers, rows):
        result = add_table(doc, headers, rows)
        for row in result.rows:
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    doc.add_heading('原图分类器对照：逻辑回归与SVM', 0)
    doc.add_paragraph(f'沿用官方完整12层冻结RETFound-Green、{patients}名患者/{images}张原图、seed42/46/43的既有划分，'
                      f'每组训练/验证/测试{counts["train"]}/{counts["val"]}/{counts["test"]}人。完全复用原图冻结特征，不改变编码器、图片、标签、'
                      '裁剪或双眼融合，不重新预训练，不增加临床输入、mask或增强。')
    doc.add_paragraph('结构：双眼RGB[B,2,3,392,392]→共享冻结编码器→[B,2,384]→'
                      '双眼均值与绝对差拼接[B,768]→训练集StandardScaler→分类器。'
                      '768是输入特征维度，不是分类器类型。原逻辑回归与线性SVM都是线性决策；'
                      'RBF-SVM通过支持向量和高斯核建立非线性边界，而不是固定的768→1神经网络层。')
    doc.add_paragraph('C网格与原逻辑回归一致：[0.001,0.01,0.1,1,10]。RBF额外搜索gamma：'
                      'scale、0.0001、0.001、0.01；类别权重balanced。所有超参数只按验证AUC选取，'
                      '并列时按预设网格顺序选择。最终SVM类型也只按验证AUC选择，平局优先线性。'
                      'RBF搜索的候选更多，验证集选择仍有过拟合风险。')
    if protocol.get('linear_solver') == 'linear_svc':
        doc.add_paragraph('线性SVM使用LinearSVC的原问题求解器（dual=False、squared_hinge、tol=1e-6），'
                          'RBF使用SVC（hinge损失）。线性SVC在本数据规模下达到迭代上限，失败结果未采用；'
                          '因此本次线性SVM与历史小数据实验的线性SVC并非完全相同的优化目标。')
    doc.add_paragraph('主结果采用各分类器的自然阈值：逻辑回归概率0.5，SVM决策分数0。'
                      'SVM分数不是概率，不能套用0.5阈值或当作概率解释。AUC直接用连续决策分数计算。'
                      '补充阈值统一只按验证集BACC选择；测试集不参与选参、选模型或调阈值。'
                      '三seed为同队列的重叠划分，样本标准差不是置信区间，观察到均值提高不代表统计显著。')
    doc.add_heading('测试集三seed均值±样本标准差', 1)
    rows = []
    for task in TASKS:
        for model in MODELS:
            metrics = summary['aggregate'][task][model]['metrics_at_default_threshold']['test']
            rows.append([tasks[task], names[model]] + [f'{metrics[k]["mean"]:.4f}±{metrics[k]["sample_std"]:.4f}'
                         for k in ('auc', 'acc', 'balanced_acc', 'f1')])
    table(['任务', '分类器', 'AUC', 'ACC', 'BACC', 'F1'], rows)
    for task in TASKS:
        for seed in SEEDS:
            doc.add_page_break()
            doc.add_heading(f'{tasks[task]} seed={seed}', 1)
            models = summary['experiments'][task][str(seed)]
            selected = models['selected_svm']
            doc.add_paragraph(f'验证集选定SVM：{selected["selected_parameters"]}；'
                              '以下混淆矩阵按[[TN,FP],[FN,TP]]排列。')
            for model in MODELS:
                r = models[model]
                doc.add_heading(names[model], 2)
                doc.add_paragraph(f'参数：{r["selected_parameters"]}；默认阈值={r["default_threshold"]}；'
                                  f'验证选择阈值={r["validation_threshold"]:.6f}')
                rows = []
                for split in SPLITS:
                    m = r['metrics_at_default_threshold'][split]
                    (tn, fp), (fn, tp) = m['confusion_matrix']
                    rows.append([split, tn, fp, fn, tp] + [f'{m[k]:.4f}' for k in ('auc', 'acc', 'f1')])
                table(['集合', 'TN', 'FP', 'FN', 'TP', 'AUC', 'ACC', 'F1'], rows)
                m = r['metrics_at_val_threshold']['test']
                doc.add_paragraph(f'验证阈值的补充测试结果：ACC={m["acc"]:.4f}，BACC={m["balanced_acc"]:.4f}，'
                                  f'F1={m["f1"]:.4f}，矩阵={m["confusion_matrix"]}')
    doc.add_page_break()
    doc.add_heading('文件与复核', 1)
    doc.add_paragraph(protocol.get('entrypoint', 'experiment_raw_svm.py') + '为实验入口。各任务/seed目录包含患者审计、'
                      '各SVM的results.json、predictions.json和scaler_and_svm.joblib。'
                      'joblib同时保存对应训练集标准化器与SVM，加载时只能使用可信文件。'
                      'selected_svm.json记录按验证集选择的模型位置，不按测试分数挑选。')
    lr_note = ('已重新拟合六组逻辑回归并与保存预测逐一核对；' if protocol.get('logistic_baseline_refitted', True)
               else '复用本次1524人已完成的六组逻辑回归，重载其分类头并逐一复现保存分数；')
    doc.add_paragraph(lr_note + '重新加载每个SVM并复算分数；'
                      '全部指标、阈值和混淆矩阵均从保存的预测复核。原实验输出不修改。')
    destination = output / '原图实验_逻辑回归与SVM对比.docx'
    doc.save(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'raw_svm_comparison_20261028')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    torch.set_num_threads(4)
    audit, released, eyes, baselines = load_comparisons()
    args.output.mkdir(parents=True)
    x = pair_features(eyes).astype(np.float64)
    protocol = {'started_at': datetime.now(timezone.utc).isoformat(), 'patients': 544, 'raw_images': 1088,
                'seeds': list(SEEDS), 'tasks': list(TASKS), 'weights': released['weights'],
                'weights_sha256': released['weights_sha256'], 'source_sha256': audit['source_sha256'],
                'feature_shape': list(x.shape), 'eye_feature_shape': list(eyes.shape),
                'fusion': 'concatenate bilateral mean and absolute difference', 'encoder_update': False,
                'augmentation': False, 'scaler_fit_split': 'train', 'C_grid': list(C_GRID),
                'rbf_gamma_grid': list(GAMMA_GRID), 'selection': 'validation AUC only; first grid candidate breaks ties',
                'kernel_selection': 'validation AUC only; linear breaks ties',
                'default_thresholds': {'logistic': .5, 'svm': 0.},
                'svm_probability_calibration': False, 'sklearn': sklearn.__version__,
                'script_sha256': sha256(__file__)}
    save_json(args.output / 'protocol.json', protocol)
    np.savez_compressed(args.output / 'frozen_features.npz', eyes=eyes, pair_features=x)
    experiments, provenance = {task: {} for task in TASKS}, []
    for task in TASKS:
        for seed in SEEDS:
            baseline = baselines[(task, seed)]
            records = baseline['records']
            labels = np.array([r['label'] for r in records])
            split = np.array([r['split'] for r in records])
            folder = args.output / f'{task}_seed{seed}'
            folder.mkdir()
            save_json(folder / 'data_audit.json', {'records': records, 'split_counts': baseline['result']['split_counts']})
            models = {'logistic': wrapped_logistic(baseline['result'])}
            fitted = {}
            for kernel in ('linear', 'rbf'):
                fit = fit_svm(x, labels, split, kernel, seed)
                name = kernel + '_svm'
                fitted[name] = fit
                result = fit['result']
                result.update(task=task, seed=seed, split_counts=baseline['result']['split_counts'],
                              weights_sha256=released['weights_sha256'], pair_feature_dim=768)
                model_dir = folder / name
                model_dir.mkdir()
                artifact = model_dir / 'scaler_and_svm.joblib'
                joblib.dump({'scaler': fit['scaler'], 'classifier': fit['model'], 'feature_description': protocol,
                             'default_threshold': 0., 'validation_threshold': result['validation_threshold']}, artifact)
                restored = joblib.load(artifact)
                reproduced = restored['classifier'].decision_function(restored['scaler'].transform(x))
                np.testing.assert_allclose(reproduced, fit['scores'], rtol=0, atol=1e-12)
                verify_predictions(records, reproduced, result)
                predictions = [{**{k: r[k] for k in ('record_id', 'patient_key', 'split', 'label')},
                                'decision_score': float(score), 'prediction_default': int(score >= 0.),
                                'prediction_val_threshold': int(score >= result['validation_threshold'])}
                               for r, score in zip(records, reproduced)]
                save_json(model_dir / 'predictions.json', predictions)
                save_json(model_dir / 'results.json', result)
                models[name] = result
                print(f'RESULT {task} seed={seed} {name}: '
                      + json.dumps(result['metrics_at_default_threshold']['test']), flush=True)
            chosen = max(('linear_svm', 'rbf_svm'), key=lambda name: models[name]['selected_validation_auc'])
            models['selected_svm'] = {**models[chosen], 'selected_kernel_model': chosen,
                                      'selection_split': 'val', 'selection_metric': 'auc'}
            save_json(folder / 'selected_svm.json', {'selected': chosen, 'selection_split': 'val',
                      'model': str((folder / chosen / 'scaler_and_svm.joblib').resolve()),
                      'validation_auc': models[chosen]['selected_validation_auc']})
            experiments[task][str(seed)] = models
            provenance.append({'task': task, 'seed': seed, 'baseline': str(baseline['folder']),
                               'results_sha256': sha256(baseline['folder'] / 'results.json'),
                               'predictions_sha256': sha256(baseline['folder'] / 'predictions.json')})
    aggregate = {task: {model: {key: {split: {metric: {
        'mean': float(np.mean([experiments[task][str(s)][model][key][split][metric] for s in SEEDS])),
        'sample_std': float(np.std([experiments[task][str(s)][model][key][split][metric] for s in SEEDS], ddof=1))}
        for metric in METRICS} for split in SPLITS}
        for key in ('metrics_at_default_threshold', 'metrics_at_val_threshold')}
        for model in MODELS} for task in TASKS}
    summary = {'protocol': protocol, 'experiments': experiments, 'aggregate': aggregate,
               'baseline_provenance': provenance}
    save_json(args.output / 'summary.json', summary)
    report = write_report(args.output, summary)
    save_json(args.output / 'verification.json', {'status': 'passed', 'baseline_heads_refitted': 6,
              'new_svm_models': 12, 'saved_models_reloaded': 12, 'patient_images_partitions_unchanged': True,
              'all_new_scores_and_confusion_matrices_reproduced': True, 'report': str(report)})
    save_json(args.output / 'status.json', {'state': 'completed', 'finished_at': datetime.now(timezone.utc).isoformat()})
    print(f'COMPLETE {report}', flush=True)


if __name__ == '__main__':
    main()
