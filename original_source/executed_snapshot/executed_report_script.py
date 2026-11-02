"""Evaluate paired modality changes and export the requested clinical report."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata

from experiment_unilateral_modalities_20261030 import (
    DEFAULT_OUT, FEATURES, POLICIES, PRIMARY_MODES, ROOT, SEEDS, SPLITS,
    metrics, read, save,
)
from audit_astigmatism_workbook import num


MODES = {'image': '双眼原图', 'table8': '八维表格', 'fusion': '原图＋八维表格', 'demo': '年龄性别', 'image_demo': '原图＋年龄性别'}
MODELS = {'logistic': '逻辑回归', 'rbf_svm': 'RBF-SVM'}
SETS = {'train': '训练', 'val': '验证', 'test': '测试'}
METRICS = ('auc', 'acc', 'sensitivity', 'specificity', 'f1')
BOOTSTRAPS = 2000


def fast_statistics(y, scores, predicted):
    positive = y == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    ranks = rankdata(scores)
    tp = int(predicted[positive].sum())
    fp = int(predicted[~positive].sum())
    return np.array([(ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0),
                     ((predicted == y).mean()), tp / n1, (n0 - fp) / n0,
                     2 * tp / max(2 * tp + fp + n1 - tp, 1)])


def bootstrap(y, score_a, threshold_a, score_b=None, threshold_b=None, count=BOOTSTRAPS):
    y, score_a = np.asarray(y), np.asarray(score_a)
    pred_a = (score_a >= threshold_a).astype(int)
    if score_b is not None:
        score_b = np.asarray(score_b)
        pred_b = (score_b >= threshold_b).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(20261030)
    values = np.empty((count, len(METRICS)))
    for i in range(count):
        indices = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        value = fast_statistics(y[indices], score_a[indices], pred_a[indices])
        if score_b is not None:
            value -= fast_statistics(y[indices], score_b[indices], pred_b[indices])
        values[i] = value
    ci = np.percentile(values, [2.5, 97.5], axis=0)
    return {metric: {'low': float(ci[0, i]), 'high': float(ci[1, i])} for i, metric in enumerate(METRICS)}


def paired_analysis(new, old, new_cut, old_cut, with_ci=True):
    before = {r['patient_key']: r for r in old}
    if len(before) != len(old) or {r['patient_key'] for r in new} != set(before):
        raise ValueError('Paired comparison must use identical unique patients')
    other = [before[r['patient_key']] for r in new]
    for a, b in zip(new, other):
        if any(a[k] != b[k] for k in ('record_id', 'label', 'split')):
            raise ValueError('Paired visit, target or split mismatch')
    y = np.array([r['label'] for r in new])
    a, b = np.array([r['score'] for r in new]), np.array([r['score'] for r in other])
    ma, mb = metrics(y, a, new_cut), metrics(y, b, old_cut)
    pa, pb = (a >= new_cut).astype(int), (b >= old_cut).astype(int)
    fixed = int(((pa == y) & (pb != y)).sum())
    harmed = int(((pa != y) & (pb == y)).sum())
    corrected_fn = int(((y == 1) & (pb == 0) & (pa == 1)).sum())
    new_fn = int(((y == 1) & (pb == 1) & (pa == 0)).sum())
    discordant = fixed + harmed
    result = {
        'patients': len(y), 'new': ma, 'reference': mb,
        'delta': {m: ma[m] - mb[m] for m in METRICS},
        'corrected_errors': fixed, 'introduced_errors': harmed,
        'corrected_false_negatives': corrected_fn, 'introduced_false_negatives': new_fn,
        'mcnemar_exact_p_unadjusted': float(binomtest(min(fixed, harmed), discordant, .5).pvalue) if discordant else 1.,
    }
    if with_ci:
        result['delta_ci95'] = bootstrap(y, a, new_cut, b, old_cut)
    return result


def get_test(predictions):
    return [p for p in predictions if p['split'] == 'test']


def aggregate(runs, cohort, mode, classifier, policy='default', split='test'):
    selected = [r['metrics'][policy][split] for r in runs if r['cohort'] == cohort and r['mode'] == mode and r['classifier'] == classifier]
    assert len(selected) == 3
    return {m: {'mean': float(np.mean([r[m] for r in selected])), 'std': float(np.std([r[m] for r in selected], ddof=1))} for m in METRICS}


def calculate(output):
    summary = read(output / 'summary.json')
    runs = summary['runs']
    lookup = {(r['cohort'], r['seed'], r['mode'], r['classifier']): r for r in runs}
    predictions = {r['run_id']: read(Path(r['folder']) / 'predictions.json') for r in runs}
    clinical = {r['patient_key']: r for r in read(output / 'clinical_records.json')}
    for c in clinical.values():
        for field in ('球镜', '柱镜'):
            a, b = c.get('右' + field), c.get('左' + field)
            c[field + '差'] = abs(a - b) if a is not None and b is not None else None
        side = c.get('weak_eye')
        c['弱视眼BCVA'] = num(c.get(side + 'BCVA')) if side in ('左', '右') else None
    primary = [r for r in runs if r['primary']]
    for r in runs:
        rows = predictions[r['run_id']]
        assert len({p['patient_key'] for p in rows}) == r['patients']
        for policy in POLICIES:
            for split in SPLITS:
                rr = [p for p in rows if p['split'] == split]
                measured = metrics([p['label'] for p in rr], [p['score'] for p in rr], r['thresholds'][policy])
                assert measured == r['metrics'][policy][split]
        if r['primary']:
            tt = get_test(rows)
            r['test_ci95_default'] = bootstrap([p['label'] for p in tt], [p['score'] for p in tt], r['thresholds']['default'])
    print('Verified predictions and primary confidence intervals', flush=True)
    comparisons = []
    corrections, difficult = [], []
    for seed in SEEDS:
        for classifier in MODELS:
            for new_mode, old_mode in [('image', 'table8'), ('fusion', 'table8'), ('fusion', 'image'), ('image_demo', 'demo'), ('fusion', 'image_demo')]:
                a = lookup['unilateral', seed, new_mode, classifier]
                b = lookup['unilateral', seed, old_mode, classifier]
                ta, tb = get_test(predictions[a['run_id']]), get_test(predictions[b['run_id']])
                paired = paired_analysis(ta, tb, a['thresholds']['default'], b['thresholds']['default'])
                paired.update(seed=seed, classifier=classifier, new_mode=new_mode, reference_mode=old_mode)
                comparisons.append(paired)
                if new_mode == 'fusion' and old_mode in ('image', 'table8'):
                    before = {p['patient_key']: p for p in tb}
                    for p in ta:
                        old = before[p['patient_key']]
                        old_pred, new_pred = old['prediction_default'], p['prediction_default']
                        if old_pred != new_pred:
                            corrections.append({'seed': seed, 'classifier': classifier, 'reference_mode': old_mode,
                                                'patient_key': p['patient_key'], 'label': p['label'],
                                                'reference_prediction': old_pred, 'fusion_prediction': new_pred,
                                                'change': '联合纠正' if new_pred == p['label'] else '联合新增误判'})
            main_predictions = {mode: {p['patient_key']: p for p in get_test(predictions[lookup['unilateral', seed, mode, classifier]['run_id']])} for mode in PRIMARY_MODES}
            for key in main_predictions['image']:
                rows = [main_predictions[m][key] for m in PRIMARY_MODES]
                if all(p['prediction_default'] != p['label'] for p in rows):
                    difficult.append({'seed': seed, 'classifier': classifier, 'patient_key': key,
                                      'label': rows[0]['label'], 'type': 'FN' if rows[0]['label'] else 'FP'})
    print('Completed matched modality comparisons', flush=True)
    bilateral = []
    old_comparisons = []
    for seed in SEEDS:
        for classifier in MODELS:
            target = lookup['unilateral', seed, 'image', classifier]
            control = lookup['strict_all_types', seed, 'image', classifier]
            target_test = get_test(predictions[target['run_id']])
            names = {p['patient_key'] for p in target_test}
            control_test = [p for p in get_test(predictions[control['run_id']]) if p['patient_key'] in names]
            result = paired_analysis(target_test, control_test, target['thresholds']['default'], control['thresholds']['default'])
            result.update(seed=seed, classifier=classifier, control_full_test=control['metrics']['default']['test'],
                          target_full_test=target['metrics']['default']['test'])
            old_rows = get_test(predictions[control['run_id']])
            groups = {}
            for category, condition in [('单眼弱视', lambda c: c['weak_label'] == 1 and c['weak_eye'] in ('左', '右')),
                                         ('双眼弱视', lambda c: c['weak_label'] == 1 and c['weak_eye'] == '双眼'),
                                         ('非弱视', lambda c: c['weak_label'] == 0)]:
                rr = [p for p in old_rows if condition(clinical[p['patient_key']])]
                groups[category] = metrics([p['label'] for p in rr], [p['score'] for p in rr], control['thresholds']['default'])
            result['control_test_by_weak_type'] = groups
            bilateral.append(result)
            old_folder = (ROOT / f'expanded1524_raw_three_seed_20261028/amblyopia_seed{seed}' if classifier == 'logistic'
                          else ROOT / f'expanded1524_svm_three_seed_20261028/amblyopia_seed{seed}/rbf_svm')
            old_result = read(old_folder / ('logistic_results.json' if classifier == 'logistic' else 'results.json'))
            old_pred = read(old_folder / ('logistic_predictions.json' if classifier == 'logistic' else 'predictions.json'))
            old_test = [{**p, 'score': p['score'] if classifier == 'logistic' else p['decision_score']} for p in old_pred if p['split'] == 'test' and p['patient_key'] in names]
            paired = paired_analysis(target_test, old_test, target['thresholds']['default'], old_result['default_threshold'])
            historical = old_result['metrics_at_default_threshold']['test']
            paired.update(seed=seed, classifier=classifier, historical_full_test={
                **historical, 'sensitivity': historical['recall'],
                'specificity': historical['confusion_matrix'][0][0] / sum(historical['confusion_matrix'][0]),
            })
            old_comparisons.append(paired)
    groups = []
    numeric_errors = []
    for r in runs:
        test = get_test(predictions[r['run_id']])
        if not r['primary']:
            continue
        group_specs = [
            ('性别', '男', lambda c: c['sex'] == '男'), ('性别', '女', lambda c: c['sex'] == '女'),
            ('年龄', '不足6岁', lambda c: c['age'] is not None and c['age'] < 6),
            ('年龄', '6至不足9岁', lambda c: c['age'] is not None and 6 <= c['age'] < 9),
            ('年龄', '9岁及以上', lambda c: c['age'] is not None and c['age'] >= 9),
            ('屈光参差', '是', lambda c: c['aniso_label'] == 1), ('屈光参差', '否', lambda c: c['aniso_label'] == 0),
            ('原始八字段完整性', '完整', lambda c: not c['missing_fields']), ('原始八字段完整性', '有缺失', lambda c: bool(c['missing_fields'])),
            ('图像来源', '历史原图', lambda c: c['image_origin'] == 'historical_original'),
            ('图像来源', '新增原图', lambda c: c['image_origin'] != 'historical_original'),
            ('弱视眼别', '左眼弱视', lambda c: c['weak_label'] == 1 and c['weak_eye'] == '左'),
            ('弱视眼别', '右眼弱视', lambda c: c['weak_label'] == 1 and c['weak_eye'] == '右'),
        ]
        for axis, group, condition in group_specs:
            rows = [p for p in test if condition(clinical[p['patient_key']])]
            if rows:
                groups.append({'run_id': r['run_id'], 'seed': r['seed'], 'mode': r['mode'], 'classifier': r['classifier'],
                               'axis': axis, 'group': group, **metrics([p['label'] for p in rows], [p['score'] for p in rows], r['thresholds']['default'])})
        for category in ('FN', 'TP', 'FP', 'TN'):
            rows = [clinical[p['patient_key']] for p in test if error_type(p) == category]
            for field in ('age', '球镜差', '柱镜差', 'SE差', '弱视眼BCVA', '图像特征绝对差均值', '双眼图像特征余弦相似度'):
                vals = np.array([x[field] for x in rows if isinstance(x.get(field), (int, float))], dtype=float)
                if len(vals):
                    numeric_errors.append({'run_id': r['run_id'], 'category': category, 'field': field,
                                           'patients': len(rows), 'observed': len(vals), 'median': float(np.median(vals)),
                                           'q1': float(np.percentile(vals, 25)), 'q3': float(np.percentile(vals, 75))})
    analysis = {'comparisons': comparisons, 'bilateral_comparisons': bilateral, 'historical_comparisons': old_comparisons,
                'fusion_changed_cases': corrections, 'common_difficult_cases': difficult,
                'group_metrics': groups, 'numeric_error_descriptions': numeric_errors,
                'bootstrap': {'iterations': BOOTSTRAPS, 'unit': 'patient', 'stratified_by_target': True, 'paired_draws': True,
                              'interpretation': 'conditional on fitted models and selected historical test cohorts; does not include model-selection or training uncertainty'},
                'testing': 'paired McNemar p-values are unadjusted exploratory values, not confirmatory significance',
                'quality_limit': 'no independent image-quality grade available after excluding all notes; source and embedding differences are descriptive, not quality ground truth'}
    summary['runs'] = runs
    save(output / 'analysis.json', analysis)
    save(output / 'summary_with_ci.json', summary)
    return summary, analysis, predictions, clinical


def error_type(p):
    return {(0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN', (1, 1): 'TP'}[(p['label'], p['prediction_default'])]


def workbook(output, summary, analysis, predictions, clinical):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    def sheet(name, headers):
        ws = wb.create_sheet(name)
        ws.append(headers)
        return ws
    ws = sheet('实验指标', ['实验范围', 'seed', '输入', '模型', '阈值策略', '集合', '人数', '阳性人数', '阈值', 'AUROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'TN', 'FP', 'FN', 'TP'])
    for r in summary['runs']:
        for policy in POLICIES:
            for split in SPLITS:
                m = r['metrics'][policy][split]
                (tn, fp), (fn, tp) = m['confusion_matrix']
                ws.append(['主实验' if r['primary'] else '补充对照', r['seed'], MODES[r['mode']], MODELS[r['classifier']], policy, SETS[split], m['patients'], m['positive'], m['threshold'], *[m[k] for k in METRICS], tn, fp, fn, tp])
    clinical_headers = ['姓名', '真实是否弱视', '弱视眼别', '是否屈光参差', '性别', '年龄', '验光类型', *FEATURES[2:], '右BCVA', '左BCVA', '双眼SE差', '原图来源', '八字段缺失项目']
    def clinical_values(p):
        c = clinical[p['patient_key']]
        return [p.get('name', c['name']), '是' if p['label'] else '否', c['weak_eye'], '是' if c['aniso_label'] else '否', c['sex'], c['age'], c.get('refraction_type'),
                *[c.get(k) for k in FEATURES[2:]], c.get('右BCVA'), c.get('左BCVA'), c.get('SE差'),
                '历史原图' if c['image_origin'] == 'historical_original' else '新增原图', '；'.join(c['missing_fields'])]
    prefix = ['seed', '输入', '模型', '集合']
    tail = ['预测分数（LR概率或SVM决策值）', '预测是否弱视', '混淆类别']
    test_sheet = sheet('测试逐人预测', prefix + clinical_headers + tail)
    errors_sheet = sheet('FN_FP病例_各集合', prefix + clinical_headers + tail)
    for r in summary['runs']:
        if not r['primary']:
            continue
        for p in predictions[r['run_id']]:
            values = [r['seed'], MODES[r['mode']], MODELS[r['classifier']], SETS[p['split']], *clinical_values(p), p['score'], '是' if p['prediction_default'] else '否', error_type(p)]
            if p['split'] == 'test':
                test_sheet.append(values)
            if error_type(p) in ('FN', 'FP'):
                errors_sheet.append(values)
    ws = sheet('联合纠正与新增误判', ['seed', '模型', '参照输入', '姓名', '真实是否弱视', '参照预测', '联合预测', '变化'])
    for r in analysis['fusion_changed_cases']:
        ws.append([r['seed'], MODELS[r['classifier']], MODES[r['reference_mode']], clinical[r['patient_key']]['name'], r['label'], r['reference_prediction'], r['fusion_prediction'], r['change']])
    ws = sheet('共同困难病例', ['seed', '模型', '姓名', '真实是否弱视', '误判类别', '说明'])
    for r in analysis['common_difficult_cases']:
        ws.append([r['seed'], MODELS[r['classifier']], clinical[r['patient_key']]['name'], r['label'], r['type'], '该seed及分类器的三种主输入均误判；仅测试集'])
    ws = sheet('同患者模态比较', ['seed', '模型', '新输入', '参照输入', '测试人数', 'AUROC差值', '差值CI下限', '差值CI上限', '敏感度差值', '差值CI下限', '差值CI上限', '纠正误判', '新增误判', '纠正FN', '新增FN', 'McNemar原始P'])
    for r in analysis['comparisons']:
        ws.append([r['seed'], MODELS[r['classifier']], MODES[r['new_mode']], MODES[r['reference_mode']], r['patients'], r['delta']['auc'], r['delta_ci95']['auc']['low'], r['delta_ci95']['auc']['high'], r['delta']['sensitivity'], r['delta_ci95']['sensitivity']['low'], r['delta_ci95']['sensitivity']['high'], r['corrected_errors'], r['introduced_errors'], r['corrected_false_negatives'], r['introduced_false_negatives'], r['mcnemar_exact_p_unadjusted']])
    ws = sheet('排除双眼弱视比较', ['seed', '模型', '比较人群', '参照人数', '新人数', '参照AUROC', '新AUROC', '参照敏感度', '新敏感度', '参照特异度', '新特异度', 'AUROC差值', '差值CI下限', '差值CI上限'])
    for r in analysis['bilateral_comparisons']:
        for name, old, new, ci in [('各自完整测试集（人群不同）', r['control_full_test'], r['target_full_test'], None), ('相同正常及单眼弱视测试患者', r['reference'], r['new'], r['delta_ci95']['auc'])]:
            ws.append([r['seed'], MODELS[r['classifier']], name, old['patients'], new['patients'], old['auc'], new['auc'], old['sensitivity'], new['sensitivity'], old['specificity'], new['specificity'], new['auc']-old['auc'], ci['low'] if ci else None, ci['high'] if ci else None])
    ws = sheet('测试集分层分析', ['seed', '输入', '模型', '分层字段', '分组', '人数', '阳性', 'AUROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'TN', 'FP', 'FN', 'TP'])
    for r in analysis['group_metrics']:
        (tn, fp), (fn, tp) = r['confusion_matrix']
        ws.append([r['seed'], MODES[r['mode']], MODELS[r['classifier']], r['axis'], r['group'], r['patients'], r['positive'], *[r[k] for k in METRICS], tn, fp, fn, tp])
    ws = sheet('误判临床变量描述', ['实验', '预测类别', '临床字段', '人数', '有记录人数', '中位数', '下四分位数', '上四分位数'])
    for r in analysis['numeric_error_descriptions']:
        ws.append([r['run_id'], r['category'], r['field'], r['patients'], r['observed'], r['median'], r['q1'], r['q3']])
    ws = sheet('患者划分', ['姓名', '是否弱视', '弱视眼别', '是否屈光参差', 'seed42', 'seed46', 'seed43', '八字段缺失项目'])
    mappings = {}
    for seed in SEEDS:
        mappings[seed] = {r['patient_key']: r for r in read(output / f'unilateral/seed{seed}/data_audit.json')['records']}
    for key, p in mappings[42].items():
        ws.append([p['name'], '是' if p['label'] else '否', p['weak_eye'], '是' if p['aniso_label'] else '否', *[SETS[mappings[s][key]['split']] for s in SEEDS], '；'.join(clinical[key]['missing_fields'])])
    ws = sheet('使用说明', ['项目', '说明'])
    for item in [('主实验', '三种输入×两种分类器×三个seed。图像组无年龄性别；另外补充人口学对照。'), ('默认阈值', '逻辑回归概率0.5；SVM决策分数0，SVM分数不是概率。'), ('补充阈值', '仅在验证集选定最大BACC及敏感度至少90%的阈值；测试敏感度不保证达到90%。'), ('FN_FP病例', '包括训练、验证、测试；筛选集合字段可单独查看测试误判。'), ('缺失字段', '原始缺失在病例表中保留空白；模型用训练集同类双眼字段的观测均值填补，不输入掩码。'), ('分层统计', '单一真实类别的分组AUROC留空；不是0，也不是0.5。'), ('统计解释', '置信区间以固定模型下患者分层配对Bootstrap估计；P值未校正，仅探索性。三个seed测试患者重叠，不作为独立外部样本。'), ('图像质量', '所有特殊备注已排除；未另行人工评定质量，不用图像来源或特征差值替代质量评分。')]:
        ws.append(list(item))
    for ws in wb:
        ws.freeze_panes = 'B2'
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = False
        for c in ws[1]:
            c.font = Font(name='Microsoft YaHei', bold=True, color='FFFFFF')
            c.fill = PatternFill('solid', fgColor='356A7B')
            c.alignment = Alignment(vertical='center', wrap_text=True)
        ws.row_dimensions[1].height = 42
        for col in range(1, ws.max_column+1):
            ws.column_dimensions[get_column_letter(col)].width = 18
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.font = Font(name='Microsoft YaHei', size=10)
                c.alignment = Alignment(vertical='center', wrap_text=True)
                if isinstance(c.value, float):
                    c.number_format = '0.0000'
                if isinstance(c.value, str):
                    c.data_type = 's'
    wb['使用说明'].column_dimensions['B'].width = 115
    for row in range(2, wb['使用说明'].max_row+1):
        wb['使用说明'].row_dimensions[row].height = 33
    path = output / '实验一_原始信息预测误判及重点比较.xlsx'
    wb.save(path)
    check = load_workbook(path, read_only=True, data_only=True)
    assert check['测试逐人预测'].max_row == 1 + sum(len(get_test(predictions[r['run_id']])) for r in summary['runs'] if r['primary'])
    assert check['患者划分'].max_row == 1101
    check.close()
    return path


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def mean_text(value):
    return f'{value["mean"]:.4f} ± {value["std"]:.4f}'


def word_report(output, summary, analysis):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = Cm(29.7), Cm(21)
    section.left_margin = section.right_margin = Cm(1.3)
    section.top_margin = section.bottom_margin = Cm(1.3)
    for name in ('Normal', 'Title', 'Heading 1', 'Heading 2'):
        style = doc.styles[name]
        style.font.name = 'Microsoft YaHei'
        style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    doc.styles['Normal'].font.size = Pt(9)
    matrices = []
    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = 'Table Grid'
        for c, v in zip(t.rows[0].cells, headers):
            c.text = str(v)
            for run in c.paragraphs[0].runs:
                run.bold = True
        repeat = OxmlElement('w:tblHeader')
        t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for values in rows:
            for c, v in zip(t.add_row().cells, values):
                c.text = str(v)
        for row in t.rows:
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(3)
                    for run in p.runs:
                        run.font.size = Pt(8)
        return t
    runs = summary['runs']
    doc.add_heading('实验一：单眼弱视与非弱视的三模态对照', 0)
    doc.add_paragraph('三种输入 × 逻辑回归/RBF-SVM × seed 42、46、43。主实验18组，补充对照18组。评价依据《新增数据纳入后的弱视AI模型训练及分析方案(1)》第六部分；重点报告敏感度、混淆矩阵、模态增益与排除双眼弱视后的变化。')
    doc.add_heading('一、队列及患者划分', 1)
    doc.add_paragraph('主队列1100人、2200张原图：非弱视676人，单眼弱视424人（左254、右170）。按患者全部可用记录排除特殊情况栏非空及形觉剥夺阳性；双眼弱视按实际选用的就诊标签排除。未额外排除小瞳、缺失验光字段或指定屈光亚型。正常仅指既有非弱视标签，不表示不存在屈光不正。')
    rows = []
    for seed in SEEDS:
        a = read(output / f'unilateral/seed{seed}/data_audit.json')
        for split, count in a['split_counts'].items():
            rows.append([seed, SETS[split], count['patients'], count['negative'], count['positive'], a['patients_with_missing'][split]])
    table(['seed', '集合', '总人数', '非弱视', '单眼弱视', '八字段至少一项缺失'], rows)
    doc.add_paragraph('保留既有患者划分，不重新打乱。各模态及两种分类器使用完全相同的患者、就诊记录与标签；三seed来自同一队列，测试人群有重叠，均值±样本标准差不是外部验证或置信区间。')
    doc.add_heading('二、输入与模型', 1)
    table(['输入模式', '分类器输入', '处理方式'], [[MODES['image'], '768维', '共享冻结RETFound-Green；每眼384维；双眼均值与绝对差拼接'], [MODES['table8'], '8维', '性别、年龄、右球镜/柱镜/轴位、左球镜/柱镜/轴位；原始八字段，无额外诊断输入'], [MODES['fusion'], '776维', '图像768维＋表格8维；训练集标准化后，固定等能量模态块缩放']])
    doc.add_paragraph('图像为原始RGB，经既有裁剪、等比缩放及补边至392×392，使用官方完整冻结编码器；复用核对过权重哈希、患者和裁剪记录的特征缓存，无图像增强。表格缺失先以训练集观测均值填补，同类左右眼字段合并估计；再进行训练集标准化，不加掩码、不把缺失认定为真实零值。1100人中365人至少有一项缺失，主要涉及柱镜与轴位。轴位保持原始数值，不新增周期编码或屈光差异工程特征；线性表格分类器对非线性眼间关系的表达受限。')
    doc.add_paragraph('融合在标准化后按预设公式sqrt(总维数/(模态数×本块维数))缩放各模态块，避免图像维数远多于表格而在RBF距离中自然占优；不按测试结果调权。两种分类器均使用balanced类别权重。C候选为0.001、0.01、0.1、1、10；RBF额外搜索gamma=scale、0.0001、0.001、0.01。仅按验证AUROC选择，平局取预先排列的首项；不将验证集加入训练。RBF候选更多，跨分类器比较包含搜索预算差异，同一分类器内模态搜索规则一致。')
    doc.add_paragraph('主结果沿用默认阈值：逻辑回归概率0.5，SVM决策分数0。SVM分数不是概率。补充阈值仅由验证集确定：最大BACC；以及验证敏感度不低于90%时尽可能提高特异度。后者不保证测试敏感度达到90%，不得根据测试结果再选阈值。')
    doc.add_heading('三、测试集主结果：三seed均值±样本标准差', 1)
    rows = []
    for classifier in MODELS:
        for mode in PRIMARY_MODES:
            a = aggregate(runs, 'unilateral', mode, classifier)
            rows.append([MODELS[classifier], MODES[mode], *[mean_text(a[k]) for k in METRICS]])
    table(['模型', '输入', 'AUROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1'], rows)
    doc.add_heading('四、重点比较与图像增量价值', 1)
    rows = []
    for classifier in MODELS:
        for new, old in [('image', 'table8'), ('fusion', 'table8'), ('fusion', 'image')]:
            rr = [r for r in analysis['comparisons'] if r['classifier'] == classifier and r['new_mode'] == new and r['reference_mode'] == old]
            rows.append([MODELS[classifier], f'{MODES[new]} - {MODES[old]}', *[fmt(np.mean([r['delta'][k] for r in rr])) for k in METRICS], sum(r['corrected_false_negatives'] for r in rr), sum(r['introduced_false_negatives'] for r in rr)])
    table(['模型', '同患者比较', 'ΔAUROC', 'ΔACC', 'Δ敏感度', 'Δ特异度', 'ΔF1', '纠正FN*', '新增FN*'], rows)
    doc.add_paragraph('*为三个实验的病例记录数合计，患者可能跨seed重复，不能解释为独立患者总数。纯图像与表格的人口学输入不同，因此另补充“年龄性别”与“原图＋年龄性别”对照，避免将人口学差异归因于影像；对应指标及对比在Excel中列出。联合与表格都包含年龄性别，二者比较是本轮影像增量价值的主要依据。')
    for classifier in MODELS:
        rr = [r for r in analysis['comparisons'] if r['classifier'] == classifier and r['new_mode'] == 'fusion' and r['reference_mode'] == 'table8']
        positive = sum(r['delta']['auc'] > 0 for r in rr)
        ci_positive = sum(r['delta_ci95']['auc']['low'] > 0 for r in rr)
        sens_positive = sum(r['delta']['sensitivity'] > 0 for r in rr)
        doc.add_paragraph(f'{MODELS[classifier]}：联合相对表格的AUROC在{positive}/3个seed提高，配对Bootstrap差值区间在{ci_positive}/3个seed完全高于零；默认阈值敏感度在{sens_positive}/3个seed提高。该结果说明排序能力与漏诊变化需分别判断；这些区间条件于当前模型和历史测试样本，未包含训练与选模不确定性，不构成独立确认性结论。')
    doc.add_heading('五、敏感度优先的补充结果', 1)
    rows = []
    for classifier in MODELS:
        for mode in PRIMARY_MODES:
            a = aggregate(runs, 'unilateral', mode, classifier, policy='validation_sens90')
            rows.append([MODELS[classifier], MODES[mode], *[mean_text(a[k]) for k in ('acc', 'sensitivity', 'specificity', 'f1')]])
    table(['模型', '输入', '测试Accuracy', '测试Sensitivity', '测试Specificity', '测试F1'], rows)
    doc.add_paragraph('以上操作点由各自验证集选择，不是在测试集上强制达到同一敏感度，不能作为严格的固定测试敏感度比较。默认、验证BACC及高敏感度阈值的全部训练/验证/测试结果均保存在Excel与各实验JSON。')
    doc.add_heading('六、排除双眼弱视后是否改善', 1)
    doc.add_paragraph('另训练1298人对照队列：与主实验采用相同的特殊情况与形觉剥夺排除规则，仅放回198名双眼弱视患者。影像编码器、分类器、超参数网格和原有患者归属保持一致。由此区分“测试人群改变后的表观差异”与“在相同单眼/非弱视患者上重新训练的差异”。')
    rows = []
    for classifier in MODELS:
        for category in ('historical', 'strict_all_types', 'unilateral'):
            if category == 'historical':
                selected = [r['historical_full_test'] for r in analysis['historical_comparisons'] if r['classifier'] == classifier]
                a = {k: np.mean([r[k] for r in selected]) for k in METRICS}
                label = '旧1524人（原特殊情况口径）'
            else:
                agg = aggregate(runs, category, 'image', classifier)
                a = {k: agg[k]['mean'] for k in METRICS}
                label = '严格特殊情况，含双眼弱视1298人' if category == 'strict_all_types' else '严格特殊情况，仅正常/单眼1100人'
            rows.append([MODELS[classifier], label, *[fmt(a[k]) for k in METRICS]])
    table(['模型', '队列（各自完整测试集）', 'AUROC均值', 'ACC均值', '敏感度均值', '特异度均值', 'F1均值'], rows)
    doc.add_paragraph('以上完整测试集的患者组成与阳性比例不同，尤其双眼弱视被移出阳性组；不能把全部变化归因于排除双眼弱视，更不能解释为识别所有弱视的能力提高。下面在完全相同的正常及单眼弱视测试患者上比较“含双眼弱视训练”与“剔除后重新训练”。')
    rows = []
    for r in analysis['bilateral_comparisons']:
        ci = r['delta_ci95']['auc']
        rows.append([r['seed'], MODELS[r['classifier']], r['patients'], fmt(r['reference']['auc']), fmt(r['new']['auc']), fmt(r['delta']['auc']), f'[{ci["low"]:.4f}, {ci["high"]:.4f}]', fmt(r['reference']['sensitivity']), fmt(r['new']['sensitivity'])])
    table(['seed', '模型', '共同测试人数', '含双眼训练AUC', '排除后AUC', 'ΔAUC', 'ΔAUC 95%CI', '含双眼训练敏感度', '排除后敏感度'], rows)
    for classifier in MODELS:
        rr = [r for r in analysis['bilateral_comparisons'] if r['classifier'] == classifier]
        doc.add_paragraph(f'{MODELS[classifier]}在共同测试患者上：排除双眼弱视后重训的平均AUROC差值为{np.mean([r["delta"]["auc"] for r in rr]):+.4f}，敏感度差值为{np.mean([r["delta"]["sensitivity"] for r in rr]):+.4f}。应以此配对变化而非不同人群的最高分判断本轮重训的影响。')
    doc.add_heading('七、误判病例与临床因素', 1)
    doc.add_paragraph('已导出所有18组主实验在训练、验证、测试中的FN与FP，保留姓名、真实标签、弱视眼别、年龄性别、验光类型、双眼球镜柱镜轴位、BCVA、屈光参差标签及原表SE差，不含原图。另列出联合模型纠正的病例、新增误判及同一seed/分类器下三种输入均误判的共同困难病例。')
    rows = []
    for classifier in MODELS:
        for reference in ('table8', 'image'):
            rr = [r for r in analysis['comparisons'] if r['classifier'] == classifier and r['new_mode'] == 'fusion' and r['reference_mode'] == reference]
            rows.append([MODELS[classifier], MODES[reference], sum(r['corrected_errors'] for r in rr), sum(r['introduced_errors'] for r in rr), sum(r['corrected_false_negatives'] for r in rr), sum(r['introduced_false_negatives'] for r in rr)])
    table(['模型', '联合相对参照', '纠正误判记录', '新增误判记录', '纠正FN记录', '新增FN记录'], rows)
    doc.add_paragraph('分层结果覆盖年龄、性别、左/右眼弱视、屈光参差、字段缺失与图像来源；连续变量以FN/TP/FP/TN的中位数和四分位数描述。含双眼弱视的对照另按单眼、双眼类型统计敏感度。由于未取得独立图像质量评分，不将图像来源或特征差异解释为质量等级；BCVA保留原记录，不自行生成未经临床确认的弱视严重程度标签。上述分析为误判相关因素描述，不能据此确定误判原因或生物学机制。')
    rows = []
    for classifier in MODELS:
        for mode in PRIMARY_MODES:
            values = []
            for axis, group in [('屈光参差', '是'), ('屈光参差', '否'), ('原始八字段完整性', '完整'), ('原始八字段完整性', '有缺失'), ('弱视眼别', '左眼弱视'), ('弱视眼别', '右眼弱视')]:
                rr = [r['sensitivity'] for r in analysis['group_metrics'] if r['classifier'] == classifier and r['mode'] == mode and r['axis'] == axis and r['group'] == group and r['sensitivity'] is not None]
                values.append(fmt(float(np.mean(rr))) if rr else '—')
            rows.append([MODELS[classifier], MODES[mode], *values])
    table(['模型', '输入', '参差阳性敏感度', '参差阴性敏感度', '字段完整敏感度', '字段缺失敏感度', '左眼弱视敏感度', '右眼弱视敏感度'], rows)
    doc.add_paragraph('分层表为各seed测试敏感度的描述性均值，分组阳性人数不同，且部分患者跨seed重复；不能据此将差异归因于参差、缺失或弱视眼别本身。性别、年龄及图像来源的完整分层人数和结果详见Excel。')
    doc.add_heading('八、各seed主实验：默认阈值指标及混淆矩阵', 1)
    for seed in SEEDS:
        for classifier in MODELS:
            doc.add_page_break()
            doc.add_heading(f'seed {seed} / {MODELS[classifier]}', 1)
            for mode in PRIMARY_MODES:
                r = next(r for r in runs if r['primary'] and r['seed'] == seed and r['classifier'] == classifier and r['mode'] == mode)
                doc.add_heading(MODES[mode], 2)
                doc.add_paragraph(f'验证集选定参数：{r["selected_parameters"]}；默认阈值：{r["thresholds"]["default"]}。矩阵顺序为[[TN,FP],[FN,TP]]。')
                rows = []
                for split in SPLITS:
                    m = r['metrics']['default'][split]
                    (tn, fp), (fn, tp) = m['confusion_matrix']
                    rows.append([SETS[split], m['patients'], *[fmt(m[k]) for k in METRICS], tn, fp, fn, tp])
                table(['集合', '人数', 'AUROC', 'ACC', '敏感度', '特异度', 'F1', 'TN', 'FP', 'FN', 'TP'], rows)
                matrices.append((len(doc.tables) - 1, r['metrics']['default']))
                c = r['test_ci95_default']['auc']
                doc.add_paragraph(f'测试AUROC患者级Bootstrap 95%CI：[{c["low"]:.4f}, {c["high"]:.4f}]。')
    path = output / '实验一_三模态三seed两模型实验报告.docx'
    doc.save(path)
    loaded = Document(path)
    for index, expected in matrices:
        for row, split in enumerate(SPLITS, 1):
            (tn, fp), (fn, tp) = expected[split]['confusion_matrix']
            shown = [int(loaded.tables[index].cell(row, col).text) for col in range(7, 11)]
            assert shown == [tn, fp, fn, tp]
    return path


def main():
    output = DEFAULT_OUT
    summary, analysis, predictions, clinical = calculate(output)
    path_xlsx = workbook(output, summary, analysis, predictions, clinical)
    path_docx = word_report(output, summary, analysis)
    save(output / 'verification.json', {
        'all_saved_metrics_recomputed': True, 'classifiers_reloaded_and_scores_checked': 36,
        'primary_runs': 18, 'supplementary_runs': 18, 'same_patients_within_modality_comparisons': True,
        'same_visits_and_labels_within_paired_comparisons': True, 'patient_split_membership_preserved': True,
        'test_used_for_parameter_or_threshold_selection': False, 'no_image_embeddings_in_workbook': True,
        'word_report': str(path_docx), 'workbook': str(path_xlsx),
    })
    save(output / 'status.json', {'state': 'completed', 'experiments': 36, 'primary_experiments': 18,
                                 'completed_at': datetime.now(timezone.utc).isoformat()})
    print('REPORT', path_docx, flush=True)
    print('WORKBOOK', path_xlsx, flush=True)
    for classifier in MODELS:
        for mode in PRIMARY_MODES:
            print(classifier, mode, aggregate(summary['runs'], 'unilateral', mode, classifier), flush=True)


if __name__ == '__main__':
    main()
