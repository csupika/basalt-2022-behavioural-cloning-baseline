import argparse
import urllib.request
import os
import glob
import cv2
import json
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Download OpenAI contractor datasets")
parser.add_argument("--json-file", type=str, required=True, help="Path to the index .json file")
parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
parser.add_argument("--num-demos", type=int, default=None, help="Maximum number of demonstrations to download")

def relpaths_to_download(relpaths, output_dir):
    def read_json(file_name):
        with open(file_name.replace('mp4', 'jsonl'), 'r') as json_file:
            text = json.loads('['+''.join(json_file.readlines()).replace('\n', ',')+']')

    data_path = '/'.join(relpaths[0].split('/')[:-1])
    non_defect=[]
    for vid_name in glob.glob(os.path.join(output_dir,'*.mp4')):
        try:
            vid = cv2.VideoCapture(vid_name)
            read_json(vid_name.replace('mp4', 'jsonl'))
            if vid.isOpened():
                non_defect.append(os.path.join(data_path, vid_name.split('/')[-1]))
        except:
            continue

    relpaths = set(relpaths)
    non_defect = set(non_defect)
    diff_to_download = relpaths.difference(non_defect)
    print('total:', len(relpaths), '| exist:', len(non_defect), '| downloading:', len(diff_to_download))
    return diff_to_download

def download_file(url, jsonl_url, outpath, jsonl_outpath, percent_done):
    print(f"[{percent_done:.0f}%] Downloading {outpath}")
    try:
        urllib.request.urlretrieve(url, outpath)
        urllib.request.urlretrieve(jsonl_url, jsonl_outpath)
    except Exception as e:
        print(f"\tError downloading {url}: {e}. Moving on\n")


def download_files(relpaths, basedir, output_dir):
    with ThreadPoolExecutor(max_workers=100) as executor:
        for i, relpath in enumerate(relpaths):
            url = basedir + relpath
            filename = os.path.basename(relpath)
            outpath = os.path.join(output_dir, filename)
            jsonl_url = url.replace(".mp4", ".jsonl")
            jsonl_filename = filename.replace(".mp4", ".jsonl")
            jsonl_outpath = os.path.join(output_dir, jsonl_filename)

            percent_done = 100 * i / len(relpaths)
            executor.submit(download_file, url, jsonl_url, outpath, jsonl_outpath, percent_done)



def main(args):
    with open(args.json_file, "r") as f:
        data = f.read()
    data = eval(data)
    basedir = data["basedir"]
    relpaths = data["relpaths"]
    if args.num_demos is not None:
        relpaths = relpaths[:args.num_demos]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    relpaths = relpaths_to_download(relpaths, args.output_dir)

    download_files(relpaths, basedir, args.output_dir)

if __name__ == "__main__":
    # args = parser.parse_args()

    Args = namedtuple('Args', ['json_file', 'num_demos', 'output_dir'])
    # args = Args(json_file="find-cave-Jul-28.json", num_demos=10, output_dir="../data/10_data/")
    args = Args(json_file="find-cave-Jul-28.json", num_demos=1500, output_dir="/mnt/data/plc2000/1500_MineRLBasaltFindCave-v0/")
    main(args)
    print("#########################< Done >#########################")
