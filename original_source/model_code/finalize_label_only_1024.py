"""Finalize the supplement by supplied labels only; do not inspect images."""
import copy
import csv
import json
from pathlib import Path

from prepare_clean_1024_split import ROOT, SEEDS, make_split, split_counts, dump


def main():
    prior = ROOT / 'model_code/outputs/splits/analysis_1024_clean_20261028'
    out = ROOT / 'model_code/outputs/splits/analysis_1024_label_only_20261028'
    if out.exists():
        raise FileExistsError(out)
    previous = json.loads((prior / 'summary.json').read_text())
    records = json.loads((ROOT / 'dataset/analysis_1024_clean_20261028/patient_manifest.json').read_text())['records']
    original = json.loads((ROOT / 'model_code/outputs/amblyopia_raw_frozen_seed42_same544/data_audit.json').read_text())['records']
    source = json.loads((ROOT / 'model_code/outputs/audits/dataset_1024_20261028_canonical/workbook_records.json').read_text())
    notes = json.loads((ROOT / 'model_code/outputs/audits/dataset_1024_20261028_canonical/issues.json').read_text())
    # The user explicitly trusts the new labels: do not exclude disagreements.
    for item in notes['changed_same_visit_labels']:
        new = next(r for r in source if r['excel_row'] == item['new_excel_row'])
        old = copy.deepcopy(next(r for r in original if r['patient_key'] == item['name']))
        old.update(weak_label=new['weak_label'], aniso_label=new['aniso_label'], label=new['weak_label'],
                   anisometropia_label=new['aniso_label'], weak_eye=new.get('原表弱视眼别') or '无',
                   weak_eye_source=new.get('原表弱视眼别') or '', image_origin='historical_original',
                   label_source='supplied new Excel labels', source_excel_row=new['excel_row'])
        records.append(old)
    records.sort(key=lambda r: r['patient_key'])
    out.mkdir(parents=True)
    counts = {}
    for seed in SEEDS:
        path = ROOT / f'model_code/outputs/splits/anisometropia_601_seed{seed}_400_100_101/image_pairs.json'
        anchors = {r['patient_key']: r['split'] for r in json.loads(path.read_text())}
        assignment = make_split(records, anchors, seed)
        folder = out / f'seed{seed}'
        folder.mkdir()
        combined = [{**r, 'split': assignment[r['patient_key']]} for r in records]
        counts[str(seed)] = split_counts(records, assignment)
        dump(folder / 'patient_manifest.json', {'records': combined, 'counts': counts[str(seed)]})
        for task, field in (('amblyopia', 'weak_label'), ('anisometropia', 'aniso_label')):
            dump(folder / f'{task}_data_audit.json', {'records': [{**r, 'label': r[field]} for r in combined],
                                                    'task': task, 'labels_are_supplied_not_recomputed': True})
        with (folder / '患者划分.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['姓名', '划分', '是否弱视', '弱视眼别', '是否屈光参差'])
            for r in combined:
                writer.writerow([r['name'], {'train': '训练集', 'val': '验证集', 'test': '测试集'}[r['split']],
                                 '是' if r['weak_label'] else '否', r['weak_eye'], '是' if r['aniso_label'] else '否'])
    result = {'processing': 'provided labels only; no new image or diagnostic checks',
              'starting_candidates': previous['starting_merged_people'],
              'special_condition_exclusions': previous['special_condition_exclusions'],
              'deprivation_positive_exclusions': previous['deprivation_positive_exclusions'],
              'unique_exclusions': previous['unique_people_excluded_by_two_rules'],
              'final_people': len(records), 'images': len(records) * 2,
              'old544_retained': sum(r['patient_key'] in {x['patient_key'] for x in original} for r in records),
              'small_pupil_excluded': False, 'general_image_quality_excluded': False,
              'new_old_label_disagreements_excluded': False, 'counts': counts}
    dump(out / 'summary.json', result)
    dump(out / 'patient_manifest.json', {'records': records})
    exclusion = json.loads((prior / 'exclusion_audit.json').read_text())
    dump(out / 'exclusions.json', {'records': exclusion['rule_exclusions']})
    previous['superseded_by'] = str(out / 'summary.json')
    previous['status'] = 'superseded_by_user_label_only_policy'
    dump(prior / 'summary.json', previous)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
