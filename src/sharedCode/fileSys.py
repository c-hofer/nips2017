import os
from collections import Iterable


def explode_home_symbol(path: str)->str:
    assert isinstance(path, str)
    if path[0] == "~":
        return os.path.expanduser("~") + path[1::]  # Maybe cross platform issue here
    else:
        return path


class FileSystemObject(object):
    """
    Class representing a generic file system object.
    """
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class FileSystemObjectCollection(object):
    """
    Generic collection of FileSystemObjects.
    """
    def __init__(self, path: str=None):
        self._content = None
        path = explode_home_symbol(path)

        if path is not None:
            if not os.path.isdir(path):
                raise ValueError("Parameter path has to be a valid directory!")

            self._get_content_by_path(path)

        if self._content is None:
            self._content = []

    def _get_content_by_path(self, path: str)->Iterable:
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._content)


class File(FileSystemObject):
    """
    Representation of a file in the filesystem.
    """
    def open(self, mode: str):
        """
        Get a file descriptor for the file.
        :param mode: mode to open file descriptor with (modes like in open(...))
        :return:
        """
        return open(self.path, mode)


class Folder(FileSystemObject):
    """
    Representation of a folder in the filesystem.
    """
    def __init__(self, path, create=False):
        path = explode_home_symbol(path)

        if path is None:
            raise ValueError("path is None!")

        if not os.path.isdir(path):
            if not create:
                raise ValueError("{} path has to be a valid existing directory!".format(path))
            else:
                os.mkdir(path)

        super().__init__(os.path.basename(path), path)

    def _get_all_direct_sub_files(self)->[File]:
        return_value = []
        for file_name, file_path in [(name, os.path.join(self.path, name)) for name in os.listdir(self.path)]:
            if os.path.isfile(file_path):
                return_value.append(File(file_name, file_path))

        return return_value

    def files(self, recursive=False, name_pred=None)->[File]:
        """
        Gives a list of all files in the folder.
        :param recursive: If true, also the files in all direct and indirect sub folders are returned.
        :param name_pred: A predicate which is applied to the name property of each file. If it is false the file
        is filtered out.
        :return: :rtype:
        """
        return_value = self._get_all_direct_sub_files()

        if recursive:
            sub_folders = self.folders(recursive=True)
            return_value += sum([folder.files(name_pred) for folder in sub_folders], [])

        if name_pred and len(return_value) > 0:
            return_value = [f for f in return_value if name_pred(f.name)]

        return return_value

    def _get_all_direct_sub_folders(self):
        return_value = []
        for file_name, dir_path in [(name, os.path.join(self.path, name)) for name in os.listdir(self.path)]:
            if os.path.isdir(dir_path):
                return_value.append(Folder(dir_path))

        return return_value

    def folders(self, recursive=False, name_pred=None)->[FileSystemObject]:
        """
        Get a list of all folders in folder.
        :param recursive: If true, also the folders in all direct and indirect sub folders are returned.
        :param name_pred: A predicate which is applied to the name property of each folder. If it is false the
        folder is filtered out.
        """
        return_value = self._get_all_direct_sub_folders()

        if recursive:
            return_value += sum([folder.folders(recursive=True, name_pred=name_pred) for folder in return_value], [])

        if name_pred and len(return_value > 0):
            return_value = [f for f in return_value if name_pred(f.name)]

        return return_value

    def content(self, recursive=False, name_filter=None)->[FileSystemObject]:
        """
        Get a list of the folder's content.
        :param recursive: If true, the recursive content of the folder is returned.
        :param name_filter: A predicate which is applied to the name property of each folder or file. If it is false the
        object is filtered out.
        :return: :rtype:
        """
        return self.folders(recursive=recursive, name_pred=name_filter) + self.files(recursive=recursive, name_pred=name_filter)


class FileCollection(FileSystemObjectCollection):
    """
    Collection which consists only of files.
    """
    def _get_content_by_path(self, path: str):
        self._content = []

        for file_name, file_path in [(name, os.path.join(path, name)) for name in os.listdir(path)]:
            if os.path.isfile(file_path):
                self._content.append(File(file_name, file_path))


class FolderCollection(FileSystemObjectCollection):
    """
    Collection which consists only of folders.
    """
    def _get_content_by_path(self, path: str) -> Iterable:
        self._content = []

        for file_name, file_path in [(name, os.path.join(path, name)) for name in os.listdir(path)]:
            if os.path.isdir(file_path):
                self._content.append(File(file_name, file_path))
