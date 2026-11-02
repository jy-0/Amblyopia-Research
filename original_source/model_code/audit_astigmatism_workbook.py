"""Read-only audit of the new annotated workbook against existing cohorts."""
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

SOURCE = Path('/root/患者信息_添加散光标注计算SE.xlsx')
ROOT = Path(__file__).resolve().parent / 'outputs'


def text(value):
    return '' if value is None else str(value).strip()


def num(value):
    try:
        value = float(text(value))
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def counts(rows, field):
    return dict(Counter(text(r.get(field)) or '[missing]' for r in rows))


def age(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:岁)?', text(value))
    return float(match.group(1)) if match else None


def status(row):
    value = text(row['原表弱视判断'])
    if value in ('确定弱视', '弱视', '是'):
        return 1
    if value == '无弱视':
        return 0
    return None


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    workbook = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    rows, sheets = [], {}
    for sheet in workbook:
        values = iter(sheet.values)
        headers = next(values)
        selected = [{**dict(zip(headers, values)), '_sheet': sheet.title, '_row': index}
                    for index, values in enumerate(values, 2) if any(v is not None for v in values)]
        sheets[sheet.title] = {'records': len(selected), 'names': len({text(r['患者姓名']) for r in selected})}
        rows.extend(selected)
    workbook.close()
    by_name = defaultdict(list)
    for row in rows:
        by_name[text(row['患者姓名'])].append(row)
    names = set(by_name) - {''}
    fields = ['性别', '年龄/检查时年龄（周岁）', '检查时间', '右眼散光轴向类型', '左眼散光轴向类型',
              '原表弱视判断', '原表弱视眼别', '原表屈光参差']
    stats = {field: counts(rows, field) for field in fields if field not in ('检查时间', '年龄/检查时年龄（周岁）')}
    missing = {k: sum(not text(r[k]) for r in rows) for k in rows[0] if not k.startswith('_')}
    duplicates = {n: [{'sheet': r['_sheet'], 'row': r['_row'], 'date': text(r['检查时间']),
                       'weak_label': text(r['原表弱视判断']), 'R': r['右眼散光轴向类型'], 'L': r['左眼散光轴向类型']}
                      for r in rs] for n, rs in by_name.items() if len(rs) > 1}
    unique_status = Counter()
    conflict_names = []
    for n, rs in by_name.items():
        labels = {status(r) for r in rs} - {None}
        unique_status['conflict' if len(labels) == 2 else 'positive' if labels == {1} else 'negative' if labels == {0} else 'uncertain'] += 1
        if len(labels) == 2:
            conflict_names.append(n)
    quality = {}
    for side in ('右', '左'):
        ccol, acol = f'{side}柱镜（原始文本）', f'{side}轴位（原始文本）'
        scol, secol, labelcol = f'{side}球镜（原始文本）', f'{side}SE（D）', f'{side}眼散光轴向类型'
        q = Counter()
        exceptions = []
        for r in rows:
            cylinder, axis, sphere, se = (num(r[k]) for k in (ccol, acol, scol, secol))
            label = r[labelcol]
            if cylinder is None:
                q['cylinder_missing_or_unparsed'] += 1
                q[f'cylinder_missing_or_unparsed_label_{label}'] += 1
            if cylinder is not None and cylinder > 0:
                q['positive_cylinder'] += 1
            if cylinder is not None and cylinder != 0 and axis is None:
                q['nonzero_cylinder_without_axis'] += 1
            if axis is not None and not 0 <= axis <= 180:
                q['axis_out_of_range'] += 1
            if sphere is not None and se is not None:
                q['se_recomputed_with_missing_cylinder_as_zero'] += 1
                if abs(se-(sphere+(cylinder or 0)/2)) > 1e-6:
                    exceptions.append({'name': r['患者姓名'], 'sheet': r['_sheet'], 'row': r['_row'], 'issue': 'SE mismatch'})
            if cylinder is not None and cylinder != 0 and axis is not None:
                q[f'{label}_axis_count'] += 1
                # This checks the workbook's apparent convention, not a clinical rule.
                inferred = '顺规' if axis <= 30 or axis >= 150 else '逆规' if 60 <= axis <= 120 else '斜轴'
                if cylinder < 0 and inferred != label:
                    q['disagree_with_30_degree_negative_cylinder_convention'] += 1
            if label == '无散光' and cylinder not in (None, 0):
                q['no_astigmatism_with_nonzero_cylinder'] += 1
        quality[side] = {'counts': dict(q), 'exceptions': exceptions}
    diff_errors, sheet_errors = [], []
    for r in rows:
        right, left, diff = (num(r[k]) for k in ('右SE（D）', '左SE（D）', '双眼SE差绝对值（D）'))
        if None in (right, left, diff) or abs(abs(right-left)-diff) > 1e-6:
            diff_errors.append([r['患者姓名'], r['_sheet'], r['_row']])
        if diff is None or (diff >= 1) != ('≥' in r['_sheet']):
            sheet_errors.append([r['患者姓名'], r['_sheet'], r['_row']])
    comparisons = {}
    current = read_csv('/root/the_dataset 1.csv')
    cohorts = {}
    for path in ('/root/the_dataset 1.csv', '/root/the_dataset.csv', '/root/dataset_20260829_reflection_complete_balanced_with_clinical.csv'):
        data = read_csv(path)
        key = 'name' if 'name' in data[0] else 'patient_key'
        cohorts[path] = data
        old_names = {text(r[key]) for r in data} - {''}
        comparisons[path] = {'records': len(data), 'unique_names': len(old_names), 'overlap_names': len(names & old_names),
                             'old_only_names': sorted(old_names-names), 'new_only_count': len(names-old_names)}
    pairs = json.loads((ROOT / 'splits/anisometropia_601_seed42_400_100_101/image_pairs.json').read_text())
    audit = json.loads((ROOT / 'amblyopia_raw_frozen_seed42_same544/data_audit.json').read_text())
    by_id = {r['record_id']: r for r in current}
    for tag, records in (('image_601', pairs), ('experiment_544', audit['records'])):
        old_names = {r['name'] for r in records}
        comparisons[tag] = {'records': len(records), 'unique_names': len(old_names), 'overlap_names': len(names & old_names),
                            'old_only_names': sorted(old_names-names), 'new_only_count': len(names-old_names)}
        summary, discrepancies, matching_details = Counter(), [], []
        for record in records:
            n = record['name']
            rs = by_name.get(n, [])
            if not rs:
                continue
            old = by_id[record['record_id']]
            summary['matched_by_name'] += 1
            summary['ambiguous_multiple_new_records'] += len(rs) > 1
            exact = [r for r in rs if all(num(r[f'{side}{field}（原始文本）']) == num(old[f'{side}{field}'])
                                        for side in ('右', '左') for field in ('球镜', '柱镜', '轴位'))]
            summary['at_least_one_exact_six_refraction_match'] += bool(exact)
            summary['unique_exact_six_refraction_match'] += len(exact) == 1
            sexes = {text(r['性别']) for r in rs}
            summary['sex_not_matching_any_candidate'] += text(old['性别']) not in sexes
            candidates = exact if exact else rs
            matched_demo = [r for r in exact if text(r['性别']) == text(old['性别'])
                            and age(r['年龄/检查时年龄（周岁）']) is not None
                            and age(r['年龄/检查时年龄（周岁）']) == age(old['年龄'])]
            summary['exact_refraction_and_sex_age_match'] += bool(matched_demo)
            matching_details.append({'name': n, 'record_id': record['record_id'],
                                     'old_date': old.get('就诊时间', ''),
                                     'exact_refraction_matches': len(exact),
                                     'exact_refraction_sex_age_matches': len(matched_demo),
                                     'candidates': [{'sheet': r['_sheet'], 'row': r['_row'], 'date': text(r['检查时间']),
                                                     'sex': r['性别'], 'age': text(r['年龄/检查时年龄（周岁）']),
                                                     'R_type': r['右眼散光轴向类型'], 'L_type': r['左眼散光轴向类型']}
                                                    for r in candidates]})
            labels = {status(r) for r in candidates} - {None}
            if labels and int(old['label_idx']) not in labels:
                summary['explicit_weak_label_disagreement'] += 1
                discrepancies.append({'name': n, 'old_label': old['label_idx'], 'new_labels': sorted(labels),
                                      'refraction_match': bool(exact), 'new_status': [r['原表弱视判断'] for r in candidates]})
            elif not labels:
                summary['new_weak_label_uncertain'] += 1
            elif len(labels) > 1:
                summary['new_weak_label_conflicting'] += 1
            else:
                summary['explicit_weak_label_agreement'] += 1
        comparisons[tag]['matching_audit'] = dict(summary)
        comparisons[tag]['weak_label_disagreements'] = discrepancies
        comparisons[tag]['record_match_details'] = matching_details
    matched601 = [r for r in pairs if r['name'] in names]
    image_stats = {'new_names_in_601': len(matched601),
                   'both_raw_files_exist': sum(all(Path(e['raw_path']).is_file() for e in r['eyes']) for r in matched601),
                   'both_reflection_files_exist': sum(all(e.get('reflection', {}).get('path') and Path(e['reflection']['path']).is_file()
                                                         for e in r['eyes']) for r in matched601)}
    result = {'source': str(SOURCE), 'sheets': sheets, 'records': len(rows), 'unique_names': len(names),
              'blank_names': sum(not text(r['患者姓名']) for r in rows), 'duplicate_names': duplicates,
              'label_distributions_records': stats, 'missing_records': missing,
              'conservative_patient_label_counts': dict(unique_status), 'conflicting_patient_labels': conflict_names,
              'conservative_policy': 'positive: 确定弱视/弱视/是; negative: 无弱视; 排除 left unresolved pending definition; grouping by name only is provisional',
              'numeric_quality': quality, 'se_difference_errors': diff_errors, 'sheet_threshold_errors': sheet_errors,
              'comparisons': comparisons, 'image_availability_in_existing_601_index': image_stats}
    output = ROOT / 'audits/astigmatism_workbook_20261014.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('duplicate_names', 'comparisons')}, ensure_ascii=False, indent=2))
    print('DUPLICATE_NAMES', len(duplicates), 'extra_rows', len(rows)-len(names))
    print('COMPARISONS', json.dumps({k: {a: b for a, b in v.items() if a not in ('weak_label_disagreements', 'record_match_details', 'old_only_names')}
                                     for k, v in comparisons.items()}, ensure_ascii=False, indent=2))
    print('Saved', output)


if __name__ == '__main__':
    main()
