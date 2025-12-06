import os
import re
import shutil

from typing import List, Union

"""
File containing useful functions dealing with the OS file system (loading, saving, navigating, etc).
"""

def recursive_match_files(dir: str, pattern: str) -> List[str]:
    """Recursively traverse subfolders of "dir" to match files using pattern.

    Returns:
        result: a list of paths to fils that match the pattern.
    """
    result = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if bool( re.search(name, pattern) ):
                result.append(os.path.join(root, name))
    return result


def find_folders_in(root: str, template: Union[str, None]=None, abs: bool=True) -> List[str]:
    """Find all folders within the root folder that match the template.

    Args:
        root: The path to the root directory.
        template: The template to match folders within root. If no template, return all.
        abs: if true, returns the absolute path. Otherwise returns 
            the folder name only.
    
    Returns:
        matching_folders: The folders matching the template.
    """
    output = []
    if not os.path.isdir(root):
        return output
    _, folders, files = next(os.walk(root))
    for f in folders:
        if abs:
            f = os.path.join(root, f)
        if template is None or folder_is_matching(f, template):
            output.append(f)
    return output


def find_files_in(root:str, template:Union[str,None]=None, abs:bool=True) -> List[str]:
    """Find all files within the root folder that match the template.

    Args:
        root: The path to the root directory.
        template: The template to match files within root. If no template, return all.
        abs: if true, returns the absolute path. Otherwise returns 
            the file name only.
    
    Returns:
        matching_files: The files matching the template.
    """
    output = []
    if not os.path.isdir(root):
        return output
    _, folders, files = next(os.walk(root))
    for f in files:
        if abs:
            f = os.path.join(root, f)
        if template is None or file_is_matching(f, template):
            output.append(f)
    return output


def file_is_matching(file: str, template: str):
    """Checks that the given folder is named as according to the template.

    The folder will be checked against the template.
    This function will return False if "folder" str is not a folder.

    Args:
        folder: the path (abs or rel) to a folder to see if named 
            according to template.
        template: the folder template to attempt to match. See FolderTemplates class

    Returns:
        match: True if the folder matches, false otherwise.
    """
    if os.path.isfile(file) or not os.path.isabs(file):
        file = os.path.basename(file)
    else:
        # Not a file
        return False

    if  bool( re.match(template, file) ):
        return True
    else:
        return False


def folder_is_matching(folder: str, template: str):
    """Checks that the given folder is named as according to the template.

    The folder will be checked against the template.
    This function will return False if "folder" str is not a folder.

    Args:
        folder: the path (abs or rel) to a folder to see if named 
            according to template.
        template: the folder template to attempt to match. See FolderTemplates class

    Returns:
        match: True if the folder matches, false otherwise. Also returns false if 
            "folder" is not a folder.
    """
    if os.path.isdir(folder) or not os.path.isabs(folder):
        folder = folder.rstrip("/")
        folder = os.path.basename(folder)
    else:
        # Not a folder
        return False

    if  bool( re.match(template, folder) ):
        return True
    else:
        return False
    

def delete_folder(folder: str, delete_self: bool=True):
    """Delete all subfolders and subfiles within a folder. If delete_self is True,
    delete the folder itself.

    Arg:
        folder: the file path to the dir
        delete_self: if true, delete the dir at "folder" as well.
    """
    # Do nothing if the folder doesn"t exist
    if not os.path.isdir(folder):
        return

    if len(os.path.abspath(folder).split("/")) < 3:
        raise RuntimeError(f"Are we dangerously deleting a root folder? > {folder}")

    # delete everything within the folder
    for files in os.listdir(folder):
        path = os.path.join(folder, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

    # delete the folder itself
    if delete_self:
        os.remove(folder)