"""Restrict the existing cohort by source notes and diagnosis; preserve splits."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from audit_dataset_1024 import norm


ROOT = Path('/root/hospital')
SOURCE = ROOT / 'model_code/outputs/splits/analysis_1024_label_only_20261028'
OUT = ROOT / 'model_code/outputs/splits/normal_unilateral_strict_special_20261030'
SEEDS = (42, 46, 43)
POSITIVE = {'是', '有', '阳性', '1', 'true', 'yes', 'positive'}
BILATERAL = {'双', '双眼', '双侧', '左右', '双眼弱视'}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def class_name(record):
    if record['weak_label'] == 0:
        return '正常（非弱视）'
    return {'左': '左眼弱视', '右': '右眼弱视'}[norm(record['weak_eye'])]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    cohort = json.loads((SOURCE / 'patient_manifest.json').read_text())['records']
    new_rows = json.loads((ROOT / 'model_code/outputs/audits/dataset_1024_20261028_canonical/workbook_records.json').read_text())
    with Path('/root/the_dataset 1.csv').open(encoding='utf-8-sig', newline='') as handle:
        old_rows = list(csv.DictReader(handle))
    evidence = defaultdict(list)
    for origin, rows, name_field, note_field, dep_field in (
        ('new_excel', new_rows, 'normalized_name', '原表特殊情况', '原表形觉剥夺因素'),
        ('historical_csv', old_rows, 'patient_key', '特殊情况', '形觉剥夺因素'),
    ):
        for row in rows:
            flags = []
            if norm(row.get(note_field)):
                flags.append('special_condition_nonempty')
            if norm(row.get(dep_field)).casefold() in POSITIVE:
                flags.append('form_deprivation_positive')
            if flags:
                evidence[norm(row[name_field])].append({
                    'source': origin, 'record_id': row.get('record_id'), 'excel_row': row.get('excel_row'),
                    'flags': flags, 'special_note': row.get(note_field), 'deprivation': row.get(dep_field),
                })
    retained, excluded = [], []
    for r in cohort:
        items = evidence.get(r['patient_key'], [])
        flags = {flag for item in items for flag in item['flags']}
        if r['weak_label'] == 1 and norm(r.get('weak_eye')) in BILATERAL:
            flags.add('bilateral_amblyopia')
        if r['weak_label'] == 1 and norm(r.get('weak_eye')) not in {'左', '右'} | BILATERAL:
            flags.add('amblyopia_laterality_not_explicit')
        if flags:
            excluded.append({'name': r['name'], 'patient_key': r['patient_key'], 'weak_eye': r.get('weak_eye'), 'reasons': sorted(flags), 'evidence': items})
        else:
            retained.append(r)
    names = {r['patient_key'] for r in retained}
    assert len(names) == len(retained) == 1100
    assert all(not evidence.get(name) for name in names)
    assert all(r['weak_label'] == 0 or norm(r['weak_eye']) in {'左', '右'} for r in retained)
    OUT.mkdir(parents=True)
    write_json(OUT / 'patient_manifest.json', {'records': retained})
    write_json(OUT / 'exclusions.json', {'starting_cohort': len(cohort), 'records': excluded})
    summary = {
        'source_cohort_patients': len(cohort), 'retained_patients': len(retained), 'retained_images': len(retained) * 2,
        'normal_means': 'provided non-amblyopia label; not a new certification of ocular health',
        'normal': sum(r['weak_label'] == 0 for r in retained),
        'unilateral_amblyopia': sum(r['weak_label'] == 1 for r in retained),
        'class_counts': dict(Counter(class_name(r) for r in retained)),
        'new_excluded_unique_patients': len(excluded),
        'new_exclusion_counts_overlapping': dict(Counter(reason for r in excluded for reason in r['reasons'])),
        'rules': {
            'special_condition': 'any nonempty provided note in any available visit; no benign-note exemptions',
            'deprivation': 'explicit positive label; negative or missing is not treated as positive',
            'bilateral': 'exclude bilateral diagnosis in the actual selected experimental visit',
            'split': 'preserve existing patient train/val/test membership; no new random split',
            'clinical_reassessment': False, 'images_reassessed': False,
        },
        'seeds': {},
    }
    split_maps = {}
    for seed in SEEDS:
        original = json.loads((SOURCE / f'seed{seed}/patient_manifest.json').read_text())['records']
        selected = [r for r in original if r['patient_key'] in names]
        assert len(selected) == len({r['patient_key'] for r in selected}) == len(retained)
        folder = OUT / f'seed{seed}'
        folder.mkdir()
        write_json(folder / 'patient_manifest.json', {'records': selected})
        for task, field in [('amblyopia', 'weak_label'), ('anisometropia', 'aniso_label')]:
            write_json(folder / f'{task}_data_audit.json', {'task': task, 'records': [{**r, 'label': r[field]} for r in selected], 'labels_are_supplied_not_recomputed': True})
        counts = {}
        for split in ('train', 'val', 'test'):
            rows = [r for r in selected if r['split'] == split]
            counts[split] = {'patients': len(rows), 'images': len(rows) * 2, 'normal': sum(not r['weak_label'] for r in rows), 'unilateral_amblyopia': sum(r['weak_label'] for r in rows), 'left_amblyopia': sum(r['weak_label'] == 1 and r['weak_eye'] == '左' for r in rows), 'right_amblyopia': sum(r['weak_label'] == 1 and r['weak_eye'] == '右' for r in rows)}
        summary['seeds'][str(seed)] = counts
        split_maps[seed] = {r['patient_key']: r['split'] for r in selected}
    wb = Workbook()
    ws = wb.active
    ws.title = '保留患者及原划分'
    ws.append(['姓名', '是否弱视', '弱视眼别', '是否屈光参差', 'seed42划分', 'seed46划分', 'seed43划分'])
    labels = {'train': '训练集', 'val': '验证集', 'test': '测试集'}
    for r in sorted(retained, key=lambda r: r['name']):
        ws.append([r['name'], '是' if r['weak_label'] else '否', r['weak_eye'], '是' if r['aniso_label'] else '否', *[labels[split_maps[seed][r['patient_key']]] for seed in SEEDS]])
    ws = wb.create_sheet('本次排除患者')
    ws.append(['姓名', '本次实验弱视眼别', '排除原因', '原始特殊情况备注'])
    reason_text = {'special_condition_nonempty': '特殊情况栏非空', 'form_deprivation_positive': '形觉剥夺阳性', 'bilateral_amblyopia': '双眼弱视', 'amblyopia_laterality_not_explicit': '弱视眼别不明确'}
    for r in sorted(excluded, key=lambda r: r['name']):
        notes = list(dict.fromkeys(str(e['special_note']) for e in r['evidence'] if norm(e.get('special_note'))))
        ws.append([r['name'], r['weak_eye'], '；'.join(reason_text[x] for x in r['reasons']), '；'.join(notes)])
    for ws in wb:
        ws.freeze_panes = 'B2'
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(name='Microsoft YaHei', color='FFFFFF', bold=True)
            cell.fill = PatternFill('solid', fgColor='356A7B')
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name='Microsoft YaHei', size=10)
                cell.alignment = Alignment(vertical='center', wrap_text=True)
        for col in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(col)].width = 22
    wb['本次排除患者'].column_dimensions['C'].width = 36
    wb['本次排除患者'].column_dimensions['D'].width = 48
    xlsx = OUT / '正常及单眼弱视患者_筛选及划分.xlsx'
    wb.save(xlsx)
    check = load_workbook(xlsx, read_only=True)
    assert check.worksheets[0].max_row == len(retained) + 1
    assert check.worksheets[1].max_row == len(excluded) + 1
    check.close()
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('XLSX', xlsx)


if __name__ == '__main__':
    main()
