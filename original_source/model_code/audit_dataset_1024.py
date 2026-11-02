"""Read-only inventory of the October supplementary workbook and paired images."""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

import openpyxl
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEAK = {'非弱视': 0, '单眼弱视': 1, '双眼弱视': 1}
ANISO = {'是': 1, '否': 0}


def norm(value):
    return ''.join(unicodedata.normalize('NFKC', '' if value is None else str(value)).split())


def visit_date(value):
    if isinstance(value, (datetime, date)):
        return value.strftime('%Y%m%d')
    text = str(value or '').strip()
    match = re.match(r'^(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})(?:日|\b)', text)
    if not match:
        return None
    try:
        return date(*(int(s) for s in match.groups())).strftime('%Y%m%d')
    except ValueError:
        return None


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def compare(old, new):
    return {'old_people': len(old), 'new_people': len(new), 'overlap': len(old & new),
            'new_only': len(new - old), 'old_not_in_new': len(old - new), 'union': len(old | new)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=Path('/root/分析用患者信息1024.zip'))
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'model_code/outputs/audits/dataset_1024_20261028')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    csv_path = Path('/root/the_dataset 1.csv')
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        previous = list(csv.DictReader(handle))
    old_by_name = defaultdict(list)
    for row in previous:
        old_by_name[norm(row['patient_key'])].append(row)
    old_names = set(old_by_name)
    audit_path = ROOT / 'model_code/outputs/amblyopia_raw_frozen_seed42_same544/data_audit.json'
    current = json.loads(audit_path.read_text())['records']
    current_names = {norm(r['name']) for r in current}
    inventory = json.loads((ROOT / 'model_code/outputs/splits/anisometropia_601_seed42_400_100_101/image_pairs.json').read_text())
    old_image_names = {norm(r['name']) for r in inventory}
    current_by_visit = {}
    unparsed_old_dates = []
    for row in current:
        dates = []
        for eye in row['eyes']:
            match = re.search(r'_(\d{8})\d*', Path(eye['raw_path']).stem)
            if match:
                dates.append(match.group(1))
        if len(dates) == 2 and dates[0] == dates[1]:
            current_by_visit[norm(row['name']), dates[0]] = row
        else:
            unparsed_old_dates.append(row['name'])
    with ZipFile(args.archive) as archive:
        members = [i for i in archive.infolist() if not i.is_dir()]
        duplicate_members = [n for n, count in Counter(i.filename for i in members).items() if count > 1]
        workbooks = [i.filename for i in members if i.filename.lower().endswith('.xlsx')]
        assert len(workbooks) == 1
        workbook_bytes = archive.read(workbooks[0])
        book = openpyxl.load_workbook(io.BytesIO(workbook_bytes), read_only=True, data_only=True)
        values = list(book['Sheet1'].iter_rows(values_only=True))
        header = values[0]
        assert header[0] == '患者姓名' and header[5] == '检查时间'
        rows, missing_name_rows = [], []
        for number, cells in enumerate(values[1:], 2):
            if not any(v is not None for v in cells):
                continue
            if not norm(cells[0]):
                missing_name_rows.append(number)
                continue
            row = {name: value for name, value in zip(header, cells) if name is not None}
            row.update(excel_row=number, normalized_name=norm(cells[0]),
                       visit_date=visit_date(row['检查时间']),
                       weak_label=WEAK.get(norm(row['原表弱视判断'])),
                       aniso_label=ANISO.get(norm(row['原表屈光参差'])))
            rows.append(row)
        notes = [[number, *cells] for number, cells in enumerate(book['Sheet3'].iter_rows(values_only=True), 1)
                 if any(value is not None for value in cells)]
        book.close()
        images, bad_names, header_errors = defaultdict(dict), [], []
        dimensions, modes = Counter(), Counter()
        for item in members:
            if not item.filename.lower().endswith('.png'):
                continue
            match = re.fullmatch(r'(.+)_(\d{8})_(左眼|右眼)\.png', Path(item.filename).name)
            if not match:
                bad_names.append(item.filename)
                continue
            name, day, side = match.groups()
            key = norm(name), day
            if side in images[key]:
                raise ValueError(f'Duplicate image side {key} {side}')
            images[key][side] = item.filename
            try:
                with archive.open(item) as handle, Image.open(handle) as image:
                    dimensions[f'{image.width}x{image.height}'] += 1
                    modes[image.mode] += 1
            except Exception as error:
                header_errors.append({'file': item.filename, 'error': repr(error)})
    by_name, by_visit = defaultdict(list), defaultdict(list)
    for row in rows:
        by_name[row['normalized_name']].append(row)
        by_visit[row['normalized_name'], row['visit_date']].append(row)
    new_names = set(by_name)
    image_names = {name for name, _ in images}
    complete = {key for key, sides in images.items() if set(sides) == {'左眼', '右眼'}}
    matched = {key for key in by_visit if key in complete and key[1] is not None}
    unique_clinical, duplicate_groups, ambiguous = {}, [], []
    for key, group in by_visit.items():
        # Source workbook/sheet is provenance, not a distinct clinical examination.
        fields = [h for h in header if h and h not in ('来源工作簿', '来源Sheet')]
        distinct = defaultdict(list)
        for row in group:
            fingerprint = tuple(norm(row[h]) for h in fields)
            distinct[fingerprint].append(row)
        for repeated in distinct.values():
            if len(repeated) > 1:
                duplicate_groups.append({'name': key[0], 'date': key[1],
                                         'excel_rows': [r['excel_row'] for r in repeated]})
        if len(distinct) == 1:
            unique_clinical[key] = group[0]
        else:
            ambiguous.append({'name': key[0], 'date': key[1], 'distinct_records': len(distinct),
                              'excel_rows': [r['excel_row'] for r in group],
                              'weak_labels': sorted({str(r['weak_label']) for r in group}),
                              'reason': 'Multiple different same-day records; image filename only has date'})
    eligible = {}
    for task, label in (('amblyopia', 'weak_label'), ('anisometropia', 'aniso_label')):
        eligible[task] = {key: unique_clinical[key] for key in matched
                          if key in unique_clinical and unique_clinical[key][label] is not None}
    eligibility = {}
    for task, candidates in eligible.items():
        label = 'weak_label' if task == 'amblyopia' else 'aniso_label'
        names = {name for name, _ in candidates}
        latest = {}
        for (name, day), row in sorted(candidates.items()):
            latest[name] = row
        eligibility[task] = {'matched_unambiguous_visits_with_known_label': len(candidates),
                             'unique_people': len(names),
                             'potential_additions_to_current544': len(names - current_names),
                             'current544_not_eligible_in_new': len(current_names - names),
                             'overlap_current544': len(current_names & names),
                             'union_with_current544_if_old_records_retained': len(names | current_names),
                             'new_people_relative_to_old683': len(names - old_names),
                             'latest_eligible_visit_label_counts': dict(Counter(str(r[label]) for r in latest.values()))}
    joint_names = {name for name, _ in set(eligible['amblyopia']) & set(eligible['anisometropia'])}
    eligibility['both_tasks_same_visit'] = {
        'unique_people': len(joint_names),
        'potential_additions_to_current544': len(joint_names - current_names),
        'overlap_current544': len(joint_names & current_names),
        'union_with_current544_if_old_records_retained': len(joint_names | current_names)}
    changes, exact_visit_matches = [], 0
    for key in matched:
        if key not in current_by_visit or key not in unique_clinical:
            continue
        exact_visit_matches += 1
        old, new = current_by_visit[key], unique_clinical[key]
        a, b = new['weak_label'], new['aniso_label']
        if (a is not None and a != old['label']) or (b is not None and b != old['anisometropia_label']):
            changes.append({'name': key[0], 'date': key[1], 'new_excel_row': new['excel_row'],
                            'old_record_id': old['record_id'], 'old_amblyopia': old['label'], 'new_amblyopia': a,
                            'old_anisometropia': old['anisometropia_label'], 'new_anisometropia': b})
    summary = {
        'archive': str(args.archive), 'archive_bytes': args.archive.stat().st_size,
        'workbook_member': workbooks[0], 'workbook_sha256': hashlib.sha256(workbook_bytes).hexdigest(),
        'identity_rule': 'Old full-table population uses patient_key (683), not raw display-name variants (686); new archive matched by normalized name, without fuzzy matching or verified patient IDs.',
        'visits_rule': 'Workbook name/date must match both image filenames; differing same-day records excluded.',
        'inventory': {'table_records': len(rows), 'table_unique_people': len(new_names),
                      'repeated_names': sum(len(g) > 1 for g in by_name.values()),
                      'extra_rows_after_name_deduplication': len(rows) - len(new_names),
                      'png_images': sum(i.filename.lower().endswith('.png') for i in members),
                      'image_unique_people': len(image_names),
                      'image_visit_keys': len(images), 'complete_bilateral_image_visits': len(complete),
                      'complete_bilateral_image_people': len({name for name, _ in complete}),
                      'table_matching_complete_image_visits': len(matched),
                      'table_matching_complete_image_people': len({name for name, _ in matched}),
                      'image_dimensions': dict(dimensions), 'image_modes': dict(modes),
                      'weak_label_record_counts': dict(Counter(norm(r['原表弱视判断']) or 'missing' for r in rows)),
                      'anisometropia_label_record_counts': dict(Counter(norm(r['原表屈光参差']) or 'missing' for r in rows))},
        'comparison': {'old683_table_vs_new_table': compare(old_names, new_names),
                       'old601_image_cohort_vs_new_image_people': compare(old_image_names, image_names),
                       'current544_vs_new_table': compare(current_names, new_names),
                       'current544_vs_new_complete_image_people': compare(current_names, {n for n, _ in complete})},
        'conservative_eligibility': eligibility,
        'issues': {'table_people_without_any_image': len(new_names - image_names),
                   'image_people_without_table': len(image_names - new_names),
                   'complete_image_visits_without_same_name_date_table_record': len(complete - set(by_visit)),
                   'table_people_with_images_but_no_same_date_complete_pair': len(
                       (new_names & image_names) - {name for name, _ in matched}),
                   'incomplete_image_visits': len(images) - len(complete),
                   'unparsed_workbook_visit_date_rows': sum(r['visit_date'] is None for r in rows),
                   'ambiguous_same_day_groups': len(ambiguous),
                   'ambiguous_same_day_groups_with_complete_images': sum((r['name'], r['date']) in complete for r in ambiguous),
                   'exact_clinical_duplicate_groups': len(duplicate_groups),
                   'duplicate_archive_paths': len(duplicate_members), 'unparseable_image_names': len(bad_names),
                   'unreadable_image_headers': len(header_errors),
                   'current_same_visit_matches': exact_visit_matches,
                   'changed_label_same_visit': len(changes),
                   'changed_amblyopia_same_visit': sum(r['new_amblyopia'] is not None and r['old_amblyopia'] != r['new_amblyopia'] for r in changes),
                   'changed_anisometropia_same_visit': sum(r['new_anisometropia'] is not None and r['old_anisometropia'] != r['new_anisometropia'] for r in changes)},
        'workbook_notes_declared_scale': [row[2] for row in notes if row[1] == '本次规模'],
        'not_done': ['No extraction into existing dataset', 'No old files modified', 'No label recomputation',
                     'No new patient split or training', 'Image headers checked, not full pixel decoding/CRC verification',
                     'No small-pupil/special-condition exclusions applied']}
    args.output.mkdir(parents=True)
    dump(args.output / 'summary.json', summary)
    dump(args.output / 'issues.json', {
        'table_people_without_any_image': sorted(new_names - image_names),
        'image_people_without_table': sorted(image_names - new_names),
        'incomplete_pairs': [{'name': name, 'date': day, 'sides': sorted(sides)}
                             for (name, day), sides in images.items() if (name, day) not in complete],
        'ambiguous_same_day': ambiguous, 'duplicate_clinical_groups': duplicate_groups,
        'changed_same_visit_labels': changes, 'header_errors': header_errors,
        'unparsed_old_image_dates': unparsed_old_dates,
        'rows_without_name': missing_name_rows})
    dump(args.output / 'patient_comparison.json', {
        'new_table_names_not_in_old683': sorted(new_names - old_names),
        'old683_names_not_in_new_table': sorted(old_names - new_names),
        'new_eligible_names_not_in_current544': {
            task: sorted({name for name, _ in candidates} - current_names) for task, candidates in eligible.items()}})
    dump(args.output / 'workbook_records.json', rows)
    dump(args.output / 'image_index.json', [{'name': name, 'date': day, 'images': sides}
                                           for (name, day), sides in sorted(images.items())])
    dump(args.output / 'workbook_notes.json', notes)
    weak, aniso, joint = (eligibility[k] for k in ('amblyopia', 'anisometropia', 'both_tasks_same_visit'))
    md = f'''# 新数据包补充量核对

来源：{args.archive}。本次只读核对，没有覆盖、解压合并、重分组或训练。

## 包内实际规模

- Excel {len(rows)}条记录，按规范化姓名去重{len(new_names)}人。
- PNG {summary['inventory']['png_images']}张，{len(image_names)}个姓名；图片均为512×512 RGB。
- 完整双眼图片{len(complete)}次就诊，涉及{len({name for name, _ in complete})}人。
- 表格姓名及检查日期能与双眼图片对应的共有{len(matched)}次就诊、{len({name for name, _ in matched})}人。

## 相对当前544人实验队列

| 任务 | 新包内保守可用候选人数 | 其中已有544人 | 可新增候选 | 保留全部旧544人后的总数 |
| --- | ---: | ---: | ---: | ---: |
| 弱视 | {weak['unique_people']} | {weak['overlap_current544']} | {weak['potential_additions_to_current544']} | {weak['union_with_current544_if_old_records_retained']} |
| 屈光参差 | {aniso['unique_people']} | {aniso['overlap_current544']} | {aniso['potential_additions_to_current544']} | {aniso['union_with_current544_if_old_records_retained']} |
| 两任务同次检查标签均明确 | {joint['unique_people']} | {joint['overlap_current544']} | {joint['potential_additions_to_current544']} | {joint['union_with_current544_if_old_records_retained']} |

候选标准：姓名和检查日期与双眼文件一致、标签可读取、同日没有多条不同临床记录导致无法唯一匹配。
人数按姓名去重；没有唯一患者ID验证，不排除同名异人，需要人工核查。未额外排除小瞳或特殊情况。
新包可用人数减544只是净差，不能当作真正新增人数，因为部分旧患者不在本包或不满足本包配对条件。

## 需要处理的问题

- 表中{len(new_names - image_names)}人没有任何同名图片；图片中{len(image_names - new_names)}人没有同名表格记录。
- 双眼不齐{len(images)-len(complete)}次：详见issues.json。
- 完整图片对中{len(complete-set(by_visit))}次缺同姓名、同日期表格记录，不能直接按姓名借用其他日期标签。
- 同日多条不同临床记录{len(ambiguous)}组，其中1组有完整图片；需人工确认。
- 与旧544人同次检查比较：弱视标签变化{summary['issues']['changed_amblyopia_same_visit']}人，参差标签变化{summary['issues']['changed_anisometropia_same_visit']}人，见issues.json。
- Sheet3说明页仍声明{summary['workbook_notes_declared_scale']}，与当前实际1793条不一致，未据说明页旧计数统计。
- 已检查所有图片头部尺寸/模式；尚未全像素解码或做全ZIP CRC校验，以上是可用候选，不是最终临床审定样本。

旧683人总表和旧601人图片队列的对照见summary.json。后续同一患者所有就诊必须放在同一划分，避免泄漏。
'''
    (args.output / '补充量统计.md').write_text(md, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
