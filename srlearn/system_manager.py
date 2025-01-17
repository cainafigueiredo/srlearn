# Copyright © 2017, 2018, 2019 Alexander L. Hayes

"""Handler for file system operations on behalf of BoostSRL."""

import pathlib
import shutil
import os

__all__ = ["reset", "FileSystem"]


def reset(soft = False, path = None):
    """Reset the FileSystem

    In some circumstances, a :class:`FileSystem` object may not properly clean up
    temporary files on object deallocation. This function performs a hard reset
    by explicitly removing the directory tree from the operating system.

    Parameters
    ----------
    soft : bool (Default: False)
        A *soft reset* reports the contents without removing.

    Returns
    -------
    results : list
        A "soft reset" returns a list of the contents.
        A "hard reset" returns an empty list.

    Notes
    -----

    Since the FileSystem object reads and writes from a location alongside
    the package, these files are also removed when the package is uninstalled.
    But uninstalling and reinstalling is overkill when a solution like this
    is available.

    Examples
    --------

    This method has a few uses. This could be used for cleaning up your
    operating system in the event of failure. It could be a shorthand
    method for triggering behavior when the data directory is/not empty.
    Or the obvious case where you need to ensure temporary files are gone.

    1. Use a "soft reset" to list contents without removing:

    .. code-block:: bash

        $ python -c "from srlearn import system_manager; print(system_manager.reset(soft=True))"
        ['data0', 'data1']

    2. Trigger conditional behavior. Here we report that the directory is empty.

    >>> from srlearn import system_manager
    >>> if not system_manager.reset(soft=True):   # doctest: +SKIP
    ...     print("Currently Empty")
    Currently Empty

    3. Use a hard reset to remove any temporary files.

    >>> from srlearn import system_manager
    >>> system_manager.reset()   # doctest: +SKIP
    []
    """
    path = pathlib.Path(__file__).parent if not path else pathlib.Path(path)
    _data = path.joinpath(FileSystem.boostsrl_data_directory)

    if not _data.exists():
        return []
    if soft:
        return os.listdir(_data)

    shutil.rmtree(_data)
    return []


class BoostSRLFiles:
    """Pointers to all file locations required by BoostSRL

    After initialization, all of these should be constant.
    """

    def __init__(self, directory) -> None:
        directory = pathlib.Path(os.path.abspath(str(directory)))
        here = pathlib.Path(__file__).parent
        self.DIRECTORY = directory
        self.BOOSTSRL_BACKEND = here.joinpath("BoostSRL.jar")
        self.BOOSTSRL_TRANFER_LEARNER_BACKEND = here.joinpath("BoostSRLTransferLearner.jar")
        # self.TREEBOOSTLER_BACKEND = here.joinpath("TreeBoostler.jar")
        # self.TRANSBOOSTLER_BACKEND = here.joinpath("models/TransBoostler/TransBoostler.jar")
        self.AUC_JAR = here
        self.TRAIN_LOG = directory.joinpath("train_output.txt")
        self.TEST_LOG = directory.joinpath("test_output.txt")
        self.TRAIN_DIR = directory.joinpath("train")
        self.TEST_DIR = directory.joinpath("test")
        self.MODELS_DIR = directory.joinpath("train/models/")
        self.BRDNS_DIR = directory.joinpath("train/models/bRDNs/")
        self.TREES_DIR = directory.joinpath("train/models/bRDNs/Trees")
        self.DOT_DIR = directory.joinpath("train/models/bRDNs/dotFiles")


class FileSystem:
    """BoostSRL File System

    BoostSRL has an implicit assumption that it has access to a file system. At runtime
    it needs to both read and write with files. This object provides a view into the
    files needed, requests files from the operating system, and (most importantly)
    prepares and cleans up the file system at allocation/de-allocation time.

    Examples
    --------

    This first example may not appear to do much, but behind the scenes it is
    creating directories for each instance of ``FileSystem``, and removing
    them upon ``exit()``.

    >>> from srlearn.system_manager import FileSystem
    >>> systems = []
    >>> for _ in range(5):
    ...     systems.append(FileSystem())

    Notes
    -----

    Ideally, each instance of a :class:`srlearn.rdn.BoostedRDN` should have its own directory
    where it can operate independently. But this can be problematic and will often
    lead to duplicated data and other problems if multiple models are learned in
    parallel on the same database.

    Another option (which may be more suited to parallel tree learning) would be to
    store data in a single location, but write the log files and models to separate
    locations.

    Attributes
    ----------
    files : :class:`enum.Enum`
        Enum providing key,value pairs for a BoostSRL database
    """

    # Prefix is the main directory that all databases will reside in.
    # In case of failure, this directory should be safe to delete.
    boostsrl_data_directory = "bsrl_data"

    def __init__(self, path = None):
        """Initialize a BoostSRL File System.

        This will create directories that are cleaned up when the instance
        is de-allocated.
        """

        path = pathlib.Path(__file__).parent if not path else pathlib.Path(path)

        # Allocate a location where data can safely be stored.
        _data = path.joinpath(FileSystem.boostsrl_data_directory)
        _allotment_number = self._allocate_space(_data)
        _directory = _data.joinpath("data" + str(_allotment_number))

        self.files = BoostSRLFiles(_directory)
        self.files.TRAIN_DIR.mkdir()
        self.files.TEST_DIR.mkdir()

    # def __del__(self):
    #     """Clean up the file system on object de-allocation."""
    #     shutil.rmtree(self.files.DIRECTORY, ignore_errors = True)

    @staticmethod
    def _allocate_space(current_directory) -> int:
        """Attempt to allocate directory `data{n}`, increment until success.

        Returns
        -------
        _postfix : int
            The number corresponding to the allocated directory.
        """
        _postfix = 0
        while True:
            _attempt = current_directory.joinpath("data" + str(_postfix))
            if not _attempt.exists():
                _attempt.mkdir(parents=True)
                break
            else:
                _postfix += 1
        return _postfix
