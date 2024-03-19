import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv


def arg_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--jobid', default=0, type=int,
                        help='slurm job id')
    args = parser.parse_args()
    return args


args = arg_parser()
json_path = os.path.join('json')
filename = os.path.join(json_path, f'{args.jobid}.json')

if os.path.exists(filename):
    fr = open(filename)
    jd = json.load(fr)
    fr.close()

    if 'results' not in jd:
        raise KeyError(f"Key 'results' not found in the loaded file {filename}.")
    entries = jd['results']
    keys = ["dataset", "baseline", "method", "dimension", "c-level", "lr", "epoch", "lambda", "seed", \
            "aps-efficiency", "aps-coverage", "daps-efficiency", "daps-coverage"]
    csv_filename = os.path.join('csv', f'{args.jobid}.csv')
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        for entry in entries:
            writer.writerow([entry[key] for key in keys])
else:
    raise FileNotFoundError(f'File {filename} not found.')
