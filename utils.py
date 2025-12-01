import os
import pandas as pd
import warnings
import requests
import zipfile
import shutil
import operator

try:
    import newspaper
    from newspaper import Article
except ModuleNotFoundError:
    raise ModuleNotFoundError("Need to install newspaper3k. Try `pip install newspaper3k`")

from common.io import find_files_in
from Newspaper_Tension.constants import DEFAULT_COLUMN_LABELS

"""
Code to import GDELT tsv file properly.
"""

def load_GDELT_tsv(in_path: str) -> pd.DataFrame:
    """
    Load the GDELT tsv file.
    """
    df = pd.read_csv(in_path, sep="\t", header=None, low_memory=False)
    num_cols = df.shape[1]
    df.columns = DEFAULT_COLUMN_LABELS[:num_cols]
    return df


def download_file_from_web(args):
    """
    Download a file from the web and save it locally.
    """
    url, local_file = args
    while not os.path.isfile(local_file):
        resp = requests.get(url=url)
        with open(local_file, "wb") as buffer:
            buffer.write(resp.content)
    return local_file


def extract_zip(args):
    """
    Extract a zip file into a folder.
    """
    file_path, temp_dir, extract_dir = args

    is_extracted = True
    extracted_files = []
    z = zipfile.ZipFile(file_path, mode="r")
    contents = z.filelist
    sizes = [c.file_size for c in contents]
    for c, s in zip(contents, sizes):
        filename = os.path.basename(c.filename)
        base = os.path.splitext(filename)[0]
        extract_file = os.path.join(extract_dir, base + ".tsv")
        extracted_files.append(extract_file)
        if os.path.isfile(extract_file) and os.path.getsize(extract_file) == s:
            continue
        else:
            is_extracted = False
    
    if is_extracted:
        return
    else:
        z.extractall(path=temp_dir)  # Extract all files from the zip.

        # Copy extracted files to a permanent location.
        for file in find_files_in(temp_dir):
            filename = os.path.basename(file)
            temp_file = os.path.join(temp_dir, filename)
            base = os.path.splitext(filename)[0]
            extract_file = os.path.join(extract_dir, base + ".tsv")
            os.rename(temp_file, extract_file)
        return True


def copy_file(args):
    """
    Copy a file.
    """
    src, dest = args
    if (os.path.isfile(dest) and os.path.getsize(dest) == os.path.getsize(src)):
        return False
    elif os.path.isfile(dest):  # If an incomplete copy was found, delete and recopy.
        os.remove(dest)
    shutil.copyfile(src, dest)
    return True


def filter_copy_tsv(args):
    """
    Filter the tsv file.
    """
    in_file, out_file, method, style, foo, filters = args
    foo = foo or identity

    if os.path.isfile(out_file):  # handle case when file already exists.
        return out_file

    filter_values = list(filters.values())
    col_inds = [DEFAULT_COLUMN_LABELS.index(c) for c in filters.keys()]
    line_value_getter = operator.itemgetter(*col_inds)

    with open(in_file, "r", encoding="utf8") as in_buffer:
        with open(out_file , "w", encoding="utf8") as out_buffer:
            for line in in_buffer:
                row_values = line_value_getter(line.split("\t"))
                if len(col_inds) == 1:
                    row_values = [row_values]
                if method == "and":
                    keep = True
                    for i in range(len(filter_values)):
                        if style == "contains" and not filter_values[i] in foo(row_values[i]):
                            keep = False; break
                        elif style == "match" and filter_values[i] != foo(row_values[i]):
                            keep = False; break
                        elif style == ">" and filter_values[i] <= foo(row_values[i]):
                            keep = False; break
                        elif style == "<" and filter_values[i] >= foo(row_values[i]):
                            keep = False; break
                elif method == "or":
                    keep = False
                    for i in range(len(filter_values)):
                        if style == "contains" and filter_values[i] in foo(row_values[i]):
                            keep = True; break
                        elif style == "match" and filter_values[i] == foo(row_values[i]):
                            keep = True; break
                        elif style == ">" and filter_values[i] > foo(row_values[i]):
                            keep = True; break
                        elif style == "<" and filter_values[i] < foo(row_values[i]):
                            keep = True; break
                if keep:
                    out_buffer.write(line)
    return out_file


def download_article(args):
    """
    Download a given article.
    """
    url, local_path, summary_path = args
    # Check to see if the file is already downloaded.
    if os.path.isfile(local_path) and os.path.isfile(summary_path):
        return local_path, summary_path
    article = Article(url)
    try:
        article.download()
        article.parse()
    except:
        warnings.warn("Unable to download an article due to client failure")
        return None, None

    document = f"# Title: {article.title}\n\n{article.text}"
    with open(local_path, "w") as buffer:
        buffer.write(document)

    # Use language model to summarize article.
    if summary_path:
        try:
            article.nlp()
        except:
            warnings.warn("Unable to save the summary")
            return local_path, None
        document = f"# Title: {article.title}\n\n{article.summary}"
        with open(summary_path, "w") as buffer:
            buffer.write(document)
    return local_path, summary_path


def identity(x):
    """
    Function that does nothing.
    """
    return x


def cast_float(x):
    """
    Function that converts input to float.
    """
    return float(x)


def cast_int(x):
    """
    Function that converts input to int.
    """
    return int(x)


def cast_float_abs(x):
    """
    Function that converst the input to an absolute value float.
    """
    return abs(float(x))