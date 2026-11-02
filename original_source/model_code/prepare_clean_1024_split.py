"""Patient-level explicit-abnormality filtering and anchored supplementary splits."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from audit_dataset_1024 import norm

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'model_code/outputs/audits/dataset_1024_20261028_canonical'
DATA = ROOT / 'dataset/analysis_1024_clean_20261028'
OUT = ROOT / 'model_code/outputs/splits/analysis_1024_clean_20261028'
BENIGN = {norm(v) for v in ('', '无', '否', '正常', '无特殊', '无特殊情况', '未见异常',
                           '无明显异常', '未知', '不详', '图片质量一般', '图像质量一般',
                           '随访数据，双眼度数相差小，')}
POSITIVE = {'是', '有', '阳性', '1', 'true', 'yes', 'positive'}
NEGATIVE_OR_UNKNOWN = {'', '否', '无', '0', 'false', 'no', 'negative', '未知', '不详', '无法判断'}
SEEDS = (42, 46, 43)
SPLITS = ('train', 'val', 'test')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def reasons(note, deprivation, refraction_type=''):
    result = []
    if norm(note) not in BENIGN:
        result.append('explicit_special_condition')
    value = norm(deprivation).casefold()
    if value in POSITIVE:
        result.append('explicit_form_deprivation')
    elif value not in NEGATIVE_OR_UNKNOWN:
        raise ValueError(f'Unreviewed form-deprivation value: {deprivation!r}')
    # Small-pupil refraction and image dimensions are intentionally not filters.
    return result


def target_sizes(n):
    train = round(n * 2 / 3)
    val = (n - train) // 2
    return {'train': train, 'val': val, 'test': n - train - val}


def make_split(records, anchors, seed):
    by_name = {r['patient_key']: r for r in records}
    if len(by_name) != len(records):
        raise ValueError('Expected one record per patient')
    sizes = target_sizes(len(records))
    assignment = {name: split for name, split in anchors.items() if name in by_name}
    capacity = {split: sizes[split] - sum(v == split for v in assignment.values()) for split in SPLITS}
    if any(n < 0 for n in capacity.values()):
        raise ValueError('Historical holdout membership exceeds target split capacity')
    pool = sorted(set(by_name) - set(assignment))
    strata = [f'{by_name[name]["weak_label"]}_{by_name[name]["aniso_label"]}' for name in pool]
    if len(pool) != sum(capacity.values()):
        raise ValueError('Capacity mismatch')
    train, holdout = train_test_split(pool, train_size=capacity['train'], random_state=seed, stratify=strata)
    holdout_strata = [f'{by_name[name]["weak_label"]}_{by_name[name]["aniso_label"]}' for name in holdout]
    val, test = train_test_split(holdout, train_size=capacity['val'], test_size=capacity['test'],
                                random_state=seed, stratify=holdout_strata)
    for split, names in (('train', train), ('val', val), ('test', test)):
        for name in names:
            assignment[name] = split
    assert Counter(assignment.values()) == sizes
    assert len(assignment) == len(records)
    assert all(assignment[name] == split for name, split in anchors.items() if name in by_name)
    for split in SPLITS:
        for label in ('weak_label', 'aniso_label'):
            assert {by_name[n][label] for n, s in assignment.items() if s == split} == {0, 1}
    return assignment


def split_counts(records, assignment):
    result = {}
    for split in SPLITS:
        rows = [r for r in records if assignment[r['patient_key']] == split]
        result[split] = {'patients': len(rows), 'images': len(rows) * 2,
                         'amblyopia_positive': sum(r['weak_label'] for r in rows),
                         'amblyopia_negative': sum(1-r['weak_label'] for r in rows),
                         'anisometropia_positive': sum(r['aniso_label'] for r in rows),
                         'anisometropia_negative': sum(1-r['aniso_label'] for r in rows),
                         'joint_labels': dict(Counter(f'{r["weak_label"]}_{r["aniso_label"]}' for r in rows))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path('/root/分析用患者信息1024.zip'))
    args = parser.parse_args()
    if DATA.exists() or OUT.exists():
        raise FileExistsError('Refusing to overwrite an existing prepared dataset or split')
    summary = json.loads((AUDIT / 'summary.json').read_text())
    assert args.archive.stat().st_size == summary['archive_bytes']
    rows = json.loads((AUDIT / 'workbook_records.json').read_text())
    image_entries = json.loads((AUDIT / 'image_index.json').read_text())
    images = {(r['name'], r['date']): r['images'] for r in image_entries}
    issues = json.loads((AUDIT / 'issues.json').read_text())
    ambiguous = {(r['name'], r['date']) for r in issues['ambiguous_same_day']}
    pending_labels = {r['name'] for r in issues['changed_same_visit_labels']}
    with Path('/root/the_dataset 1.csv').open(encoding='utf-8-sig', newline='') as handle:
        historical = list(csv.DictReader(handle))
    baseline = json.loads((ROOT / 'model_code/outputs/amblyopia_raw_frozen_seed42_same544/data_audit.json').read_text())['records']
    old_names = {r['patient_key'] for r in baseline}
    evidence = defaultdict(list)
    for r in rows:
        flags = reasons(r.get('原表特殊情况'), r.get('原表形觉剥夺因素'), r.get('验光类型'))
        if flags:
            evidence[r['normalized_name']].append({'source': 'new_excel', 'excel_row': r['excel_row'],
                'reasons': flags, 'special_note': r.get('原表特殊情况'), 'deprivation': r.get('原表形觉剥夺因素')})
    for r in historical:
        flags = reasons(r.get('特殊情况'), r.get('形觉剥夺因素'), r.get('验光类型'))
        if flags:
            evidence[norm(r['patient_key'])].append({'source': 'old_csv', 'record_id': r['record_id'],
                'reasons': flags, 'special_note': r.get('特殊情况'), 'deprivation': r.get('形觉剥夺因素')})
    latest = {}
    for r in sorted(rows, key=lambda r: (r['visit_date'], str(r['检查时间']), r['excel_row'])):
        key = r['normalized_name'], r['visit_date']
        pair = images.get(key, {})
        if key not in ambiguous and set(pair) == {'左眼', '右眼'} and r['weak_label'] is not None and r['aniso_label'] is not None:
            latest[r['normalized_name']] = r
    assert len(latest) == summary['conservative_eligibility']['both_tasks_same_visit']['unique_people']
    pool = old_names | set(latest)
    assert len(pool) == 1599
    flagged = set(evidence) & pool
    pending = pending_labels & (pool - flagged)
    candidates = []
    for old in baseline:
        name = old['patient_key']
        if name in flagged or name in pending:
            continue
        r = copy.deepcopy(old)
        r.update(weak_label=old['label'], aniso_label=old['anisometropia_label'],
                 image_origin='historical_original', weak_eye=old.get('weak_eye_source') or ('无' if old['label'] == 0 else '未明确'))
        candidates.append(r)
    for name in sorted(set(latest) - old_names - flagged - pending):
        row = latest[name]
        pair = images[name, row['visit_date']]
        candidates.append({'patient_key': name, 'name': name, 'record_id': f'1024:Sheet1:{row["excel_row"]}',
                           'source_excel_row': row['excel_row'], 'visit_date': row['visit_date'],
                           'weak_label': row['weak_label'], 'aniso_label': row['aniso_label'],
                           'weak_eye': row.get('原表弱视眼别') or ('无' if row['weak_label'] == 0 else '未明确'),
                           'image_origin': 'supplement_provided_RGB',
                           'eye_order_status': 'provided_filename_not_reverified_in_pixels',
                           'eyes': [{'side': side, 'archive_member': pair[chinese]}
                                    for side, chinese in (('R', '右眼'), ('L', '左眼'))]})
    technical, fingerprints, checked, dimensions = [], defaultdict(set), [], Counter()
    with ZipFile(args.archive) as archive:
        workbook = archive.read(summary['workbook_member'])
        assert hashlib.sha256(workbook).hexdigest() == summary['workbook_sha256']
        for index, record in enumerate(candidates, 1):
            name, record_hashes = record['patient_key'], []
            try:
                for eye in record['eyes']:
                    data = (archive.read(eye['archive_member']) if 'archive_member' in eye
                            else Path(eye['raw_path']).read_bytes())
                    digest = hashlib.sha256(data).hexdigest()
                    if 'raw_sha256' in eye and digest != eye['raw_sha256']:
                        raise ValueError('Historical original image hash changed')
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        pixels = np.asarray(image.convert('RGB'))
                        if float(pixels.std()) < 1.:
                            raise ValueError('Blank or nearly constant image')
                        size = list(image.size)
                    pixel_digest = hashlib.sha256(str(size).encode() + pixels.tobytes()).hexdigest()
                    eye.update(raw_sha256=digest, raw_size=size, pixel_sha256=pixel_digest)
                    if 'archive_member' in eye:
                        eye.update(raw_path=str(DATA / 'images' / Path(eye['archive_member']).name),
                                   crop_box=[0, 0, *size])
                    fingerprints[pixel_digest].add(name)
                    record_hashes.append(pixel_digest)
                    dimensions[f'{size[0]}x{size[1]}'] += 1
                if len(set(record_hashes)) != 2:
                    raise ValueError('Same pixels supplied as both eyes')
                checked.append(record)
            except Exception as error:
                technical.append({'name': name, 'error': repr(error)})
            if index % 200 == 0 or index == len(candidates):
                print(f'CHECKED images: {index}/{len(candidates)} patients', flush=True)
        duplicate_people = set().union(*(names for names in fingerprints.values() if len(names) > 1)) if fingerprints else set()
        records = sorted((r for r in checked if r['patient_key'] not in duplicate_people), key=lambda r: r['patient_key'])
        assert len(records) == len({r['patient_key'] for r in records})
        DATA.mkdir(parents=True)
        (DATA / 'images').mkdir()
        (DATA / '患者信息总结.xlsx').write_bytes(workbook)
        for record in records:
            for eye in record['eyes']:
                if 'archive_member' in eye:
                    target = Path(eye['raw_path'])
                    assert not target.exists()
                    data = archive.read(eye['archive_member'])
                    assert hashlib.sha256(data).hexdigest() == eye['raw_sha256']
                    target.write_bytes(data)
    OUT.mkdir(parents=True)
    all_counts, anchor_counts = {}, {}
    for seed in SEEDS:
        historical_split = ROOT / f'model_code/outputs/splits/anisometropia_601_seed{seed}_400_100_101/image_pairs.json'
        previous = json.loads(historical_split.read_text())
        anchors = {r['patient_key']: r['split'] for r in previous}
        assignment = make_split(records, anchors, seed)
        folder = OUT / f'seed{seed}'
        folder.mkdir()
        combined = [{**r, 'split': assignment[r['patient_key']]} for r in records]
        all_counts[str(seed)] = split_counts(records, assignment)
        anchor_counts[str(seed)] = sum(r['patient_key'] in anchors for r in records)
        dump(folder / 'patient_manifest.json', {'records': combined, 'counts': all_counts[str(seed)],
            'seed': seed, 'historical_split_source': str(historical_split),
            'historical_split_sha256': hashlib.sha256(historical_split.read_bytes()).hexdigest()})
        for task, field in (('amblyopia', 'weak_label'), ('anisometropia', 'aniso_label')):
            dump(folder / f'{task}_data_audit.json', {'records': [{**r, 'label': r[field]} for r in combined],
                'task': task, 'source_archive': str(args.archive), 'same_cohort_both_tasks': True})
        with (folder / '患者划分.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['姓名', '划分', '是否弱视', '弱视眼别', '是否屈光参差'])
            for r in combined:
                writer.writerow([r['name'], {'train': '训练集', 'val': '验证集', 'test': '测试集'}[r['split']],
                                 '是' if r['weak_label'] else '否', r['weak_eye'], '是' if r['aniso_label'] else '否'])
    clean_new_names = (set(latest) - flagged - pending)
    final = {'source_archive': str(args.archive), 'starting_merged_people': len(pool),
             'special_condition_exclusions': sum(any('explicit_special_condition' in x['reasons'] for x in evidence[n]) for n in flagged),
             'deprivation_positive_exclusions': sum(any('explicit_form_deprivation' in x['reasons'] for x in evidence[n]) for n in flagged),
             'unique_people_excluded_by_two_rules': len(flagged),
             'pending_label_conflicts': sorted(pending), 'technical_failures': technical,
             'cross_patient_exact_duplicate_people': sorted(duplicate_people),
             'final_people': len(records), 'final_images': len(records) * 2,
             'retained_old544_people': sum(r['patient_key'] in old_names for r in records),
             'newly_added_to_old544': sum(r['patient_key'] not in old_names for r in records),
             'new_package_only_after_rule_and_conflict_filter_before_pixel_check': len(clean_new_names),
             'pixel_dimensions_checked_before_final_exclusions': dict(dimensions),
             'tasks_share_identical_people_images_and_splits': True,
             'filter_policy': 'any visit across new workbook and old CSV; explicit special abnormality OR explicit positive form deprivation',
             'not_excluded': ['small-pupil refraction', 'small image dimensions', '图片质量一般', '图像质量一般',
                              'follow-up note with small refractive difference', 'unknown or blank deprivation'],
             'special_note_benign_exact_values_normalized': sorted(BENIGN),
             'record_selection': 'retain original old544 examination/images; for added patients choose latest valid paired same-day exam',
             'label_conflicts': 'two previously identified same-name/date disagreements held pending confirmation; not relabeled',
             'ratio': 'approximately 4:1:1', 'counts': all_counts, 'historical_601_membership_retained': anchor_counts,
             'verification': {'full_pixel_decode': True, 'supplement_selected_members_crc_checked': True,
                              'historical_image_sha256_checked': True, 'patient_overlap_across_splits': 0,
                              'duplicate_pixel_images_across_splits': 0, 'old_data_unmodified': True}}
    dump(OUT / 'summary.json', final)
    dump(OUT / 'exclusion_audit.json', {'rule_exclusions': [{'name': n, 'evidence': evidence[n]} for n in sorted(flagged)],
                                       'pending_labels': sorted(pending), 'technical_failures': technical,
                                       'duplicate_image_patients': sorted(duplicate_people)})
    dump(DATA / 'patient_manifest.json', {'records': records, 'source_archive': str(args.archive), 'split_root': str(OUT)})
    lines = ['# 1024补充后可用原图数据集与划分', '',
             '保留小瞳、小尺寸图片及“图片/图像质量一般”；按患者整名排除任次特殊情况明确异常或形觉剥夺明确阳性。',
             '特殊情况中的随访/双眼度数差小只作为信息保留；其他已记录的病变、质量差/糊、疑似异常、筛选排除/不分析等剔除。',
             f'合并候选{len(pool)}人，两条规则共排除{len(flagged)}人（两项人数可能重叠）。',
             f'标签冲突待核实{len(pending)}人；技术检查失败{len(technical)}人；跨患者精确重复图片涉及{len(duplicate_people)}人。',
             f'最终{len(records)}人、{2*len(records)}张图；旧544人保留{final["retained_old544_people"]}人，新增{final["newly_added_to_old544"]}人。',
             '', '| seed | 集合 | 人数 | 图片 | 弱视 | 非弱视 | 参差 | 非参差 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed, counts in all_counts.items():
        for split, count in counts.items():
            lines.append('| ' + ' | '.join(map(str, [seed, split, count['patients'], count['images'],
                         count['amblyopia_positive'], count['amblyopia_negative'], count['anisometropia_positive'],
                         count['anisometropia_negative']])) + ' |')
    lines += ['', '两任务使用完全相同患者和图片；主划分seed42，另保留46/43作后续稳定性验证。',
              '既有601人图片队列中的保留患者沿用各seed原train/val/test；其余新增者按弱视×参差联合标签分层补足4:1:1。',
              '一名患者只选一次检查，双眼同组，同一患者不跨集合。不重新训练模型、不改旧实验结果。',
              '新提供照片为512×512 RGB，按文件名眼别配对，未重新人工核对图内L/R；旧患者保留原高分辨率照片及审计裁剪框。',
              '受姓名身份匹配限制，同名异人仍需临床ID核实。两名标签冲突患者另列排除审计，等待确认后才能补入。']
    (OUT / '数据统计与划分.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
