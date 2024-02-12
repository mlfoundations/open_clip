import csv
import json
import os


def gather_results(fnames: list[str], output: str):
    """
    Build a single benchmark CSV file from a set of JSON evaluation files
    """
    rows = []
    fieldnames = set()

    def _process_file(_path: str):
        data = json.load(open(_path))
        _row = {}
        _row.update(data['metrics'])
        _row.update(data)
        del _row['metrics']
        _row['model_fullname'] = _row['model'] + ' ' + _row['pretrained']
        for field in _row.keys():
            fieldnames.add(field)
        rows.append(_row)

    for path in fnames:
        if os.path.isdir(path):
            files = [
                os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')
            ]
            for file in files:
                _process_file(file)
        else:
            _process_file(path)

    with open(output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
