import os
import requests
import lxml.html as lh
import tempfile
import tqdm
import operator
import multiprocessing as mp
from typing import Dict, List, Callable
import warnings
import zipfile

from common.io import find_files_in

from Newspaper_Tension.constants import DEFAULT_COLUMN_LABELS, DEFAULT_DOWNLOAD_DIR, DEFAULT_GDELT_BASE_URL
from Newspaper_Tension.utils import download_article, download_file_from_web, extract_zip, filter_copy_tsv, identity, cast_float_abs


"""
Retrieve the GDELT data.

Based on https://github.com/t4f1d/sentiment-analysis/blob/master/1.%20Retrieve%20Data/RETRIEVE%20DATA.ipynb
"""


def download_helper(actor_one: str, actor_two: str, out_dir: str, subsample: int = None):
    """
    Download articles where Actor1Name = actor_one, and Actor2Name = actor_two.

    Args:
        subsample: only download every nth article.
    """
    gdelt = Gdelt_Downloader(output_dir=out_dir)
    gdelt.download_compressed()
    gdelt.extract_compressed()
    gdelt.filter_tsvs(match_style="match", filters={"Actor1Name": actor_one, "Actor2Name": actor_two, "IsRootEvent": "1"})
    gdelt.filter_tsvs(match_style=">", method="or", foo=cast_float_abs, filters={"AvgTone": 20.0})
    gdelt.download_articles(subsample=subsample)


class Gdelt_Downloader(object):
    """
    Class which downloads data from the GDELT 1.0 Dataset.
    """
    def __init__(self, base_url: str = None, output_dir: str = None, work_dir: str = None):
        """
        Constructor for Gdelt_Downloader.
        """
        self._base_url = base_url or DEFAULT_GDELT_BASE_URL
        self._output_dir = output_dir or DEFAULT_DOWNLOAD_DIR
        self._work_dir = work_dir or "/data_drive/tmp/GDELT/"
        self._compressed_dir = os.path.join(self._work_dir, "compressed")
        self._extracted_dir = os.path.join(self._work_dir, "extracted")
        self._filtered_dir = os.path.join(self._output_dir, "tsv_files")
        self._articles_dir = os.path.join(self._output_dir, "articles")
        self._summaries_dir = os.path.join(self._output_dir, "summaries")
        folders = [
            self._compressed_dir, self._extracted_dir, self._filtered_dir, self._articles_dir, 
            self._summaries_dir
        ]
        # Create all relevant folders.
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

        self._compressed_files = []
        self._tsv_files = []
        self._was_filtered = False

    def download_compressed(self, filenames: List[str] = None, num_workers: int = 8):
        """
        Download compressed csv files from the GDELT website.
        """
        # Obtain downloadable files from GDELT.
        if filenames:
            remote_files = filenames
        else:
            page = requests.get(os.path.join(self._base_url, "index.html"))
            doc = lh.fromstring(page.content)
            link_list = doc.xpath("//*/ul/li/a/@href")
            remote_files = [x for x in link_list if str.isdigit(x[0:4])]

        # Prepare arguments.
        file_urls = [os.path.join(self._base_url, file) for file in remote_files]
        local_paths = [os.path.join(self._compressed_dir, file) for file in remote_files]
        args = list(zip(file_urls, local_paths))

        # Multiprocess download.
        with mp.Pool(processes=num_workers) as pool:
            iter = tqdm.tqdm(pool.imap_unordered(download_file_from_web, args, chunksize=10),
                             total=len(args), desc="Downloading Compressed Files...",)
            self._compressed_files = list(iter)

    def extract_compressed(self, num_workers: int = 8):
        """
        Extract compressed csv files.
        """
        # Prepare arguments.
        source_files = []
        temp_dirs = []
        temp_dirnames = []
        extract_dirs = []
        for compressed in tqdm.tqdm(self._compressed_files, desc="Scanning Compressed Files..."):
            try:
                z = zipfile.ZipFile(file=compressed, mode="r")
            except:
                warnings.warn(f"{compressed} is not a zip file.")
                continue
            else:
                z.close()
            source_files.append(compressed)
            temp_dirs.append(tempfile.TemporaryDirectory(prefix=self._work_dir))
            temp_dirnames.append(temp_dirs[-1].name)
            extract_dirs.append(self._extracted_dir)
        args = list(zip(source_files, temp_dirnames, extract_dirs))

        # Pool the extraction process.
        with mp.Pool(processes=num_workers) as pool:
            iter = tqdm.tqdm(pool.imap_unordered(extract_zip, args, chunksize=10), total=len(args), 
                             desc="Extracting Compressed Files...")
            list(iter)
        
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    def filter_tsvs(
            self, method: str = "and", match_style: str = "contains", filters: Dict = None, 
            foo: Callable = None, num_workers: int = 8
        ):
        """
        Filter the tsv files by rows.
        """
        foo = foo or identity
        # Prepare the arguments.
        source_tsv_dir = self._filtered_dir if self._was_filtered else self._extracted_dir
        backups = []
        args = []
        for source in find_files_in(source_tsv_dir):
            if self._was_filtered:
                file_split = os.path.splitext(source)
                backup_file = file_split[0] + "_1" + file_split[1]
                os.rename(source, backup_file)
                backups.append(backup_file)
                source_tsv = backup_file
                copy_tsv = os.path.join(self._filtered_dir, os.path.basename(source))
            else:
                source_tsv = source
                copy_tsv = os.path.join(self._filtered_dir, os.path.basename(source))
            args.append((source_tsv, copy_tsv, method, match_style, foo, filters))
        
        # Multiprocess filtering
        with mp.Pool(processes=num_workers) as pool:
            iter = tqdm.tqdm(pool.imap_unordered(filter_copy_tsv, args, chunksize=10), 
                             total=len(args), desc="Filtering TSV files...",)
            self._filtered_files = list(iter)

        if self._was_filtered:
            for backup in backups:
                os.remove(backup)
        self._was_filtered = True

    def download_articles(self, subsample: int = None, num_workers: int = 8):
        """
        Download articles referenced in the tsv files.

        Args:
            subsample: if given an integer (n), download every nth article only.
        """
        n = subsample or 1
        args = []
        col_inds = [DEFAULT_COLUMN_LABELS.index(c) for c in ["SOURCEURL", "GLOBALEVENTID", "GoldsteinScale"]]
        getter = operator.itemgetter(*col_inds)
        for tsv in tqdm.tqdm(self._filtered_files, desc="Finding articles..."):
            with open(tsv, "r") as in_buffer:
                i = 0
                for line in in_buffer:
                    splits = line.split("\t")
                    if len(splits) < len(DEFAULT_COLUMN_LABELS):
                        continue
                    data = getter(splits)
                    if not all([str(val) for val in data]):
                        continue
                    i += 1
                    if i % n != 0:  # Download every n-th article only. Decreases download amount.
                        continue
                    url, event_id, _ = data
                    save_path = os.path.join(self._articles_dir, event_id + ".txt")
                    sum_path = os.path.join(self._summaries_dir, event_id + ".txt")
                    if os.path.isfile(save_path) and os.path.isfile(sum_path):
                        continue
                    else:
                        args.append((url, save_path, sum_path))
        
        with mp.Pool(processes=num_workers) as pool:
            iter = tqdm.tqdm(pool.imap_unordered(download_article, args, chunksize=10), 
                             total=len(args), desc="Downloading articles...")
            successes = list(iter)
        
        
if __name__=="__main__":
    """
    Execute the downloading procedure.
    """
    pairs = [
        ("RUSSIA", "UKRAINE"),
        ("UNITED STATES", "RUSSIA"),
        ("ISRAEL", "PALESTINE"),
        ("UNITED STATES", "AFGHANISTAN"),
        ("UNITED STATES", "CHINA"),
        ("UNITED STATES", "IRAQ"),
        ("CHINA", "TAIWAN"),
        ("SAUDI ARABIA", "IRAN"),
        ("UNITED STATES", "IRAN"),
    ]

    subsample_val = 10
    
    for actone, acttwo in pairs:
        print(f"Working on {actone} -> {acttwo}")
        download_helper(
            actone, acttwo, 
            os.path.join("/data_drive", "GDELT", f"{actone}_{acttwo}".replace(" ", "-")),
            subsample=subsample_val
        )
        print(f"Working on {acttwo} -> {actone}")
        download_helper(
            acttwo, actone, 
            os.path.join("/data_drive", "GDELT", f"{acttwo}_{actone}".replace(" ", "-")),
            subsample=subsample_val
        )