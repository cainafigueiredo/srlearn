# Copyright © 2017, 2018, 2019 Alexander L. Hayes

"""
Base class for Boosted Relational Models
"""

import os
from collections import Counter
import inspect
import json
import shutil
import logging
import warnings

from glob import glob
from graphviz import Source
from sklearn.utils.validation import check_is_fitted
import subprocess

from .system_manager import FileSystem
from .utils._parse_trees import parse_tree
from ._meta import __version__


warnings.simplefilter("default")


class BaseBoostedRelationalModel:
    """Base class for deriving boosted relational models

    This class extends :class:`sklearn.base.BaseEstimator` and
    :class:`sklearn.base.ClassifierMixin` while providing several utilities
    for instantiating a model and performing learning/inference with the
    BoostSRL jar files.

    .. note:: This is not a complete treatment of *how to derive estimators*.
        Contributions would be appreciated.

    Examples
    --------

    The actual :class:`srlearn.rdn.BoostedRDNClassifier` is derived from this class, so this
    example is similar to the implementation (but the actual implementation
    passes model parameters instead of leaving them with the defaults).
    This example derives a new class ``BoostedRDNClassifier``, which inherits the default
    values of the superclass while also setting a 'special_parameter' which
    may be unique to this model.

    All that remains is to implement the specific cases of ``fit()``,
    ``predict()``, and ``predict_proba()``.
    """

    def __init__(
        self,
        *,
        n_estimators=10,
        node_size=2,
        max_tree_depth=3,
        neg_pos_ratio=2,
        number_of_clauses=8,
        number_of_cycles=100,
        solver = None,
        path = None
    ):
        """Initialize a BaseEstimator"""
        self.n_estimators = n_estimators
        self.neg_pos_ratio = neg_pos_ratio
        self.node_size = node_size
        self.max_tree_depth = max_tree_depth
        self.number_of_clauses = number_of_clauses
        self.number_of_cycles = number_of_cycles
        self.path = path

        if solver is None:
            self.solver = "BoostSRL"
        else:
            if solver not in ("BoostSRL", "BoostSRLTransferLearner", "TreeBoostler", "TransBoostler"):
                raise ValueError("`solver` must be 'BoostSRL', 'BoostSRLTransferLearner', 'TreeBoostler', or 'TransBoostler'")
            self.solver = solver

    @classmethod
    def _get_param_names(cls):
        # Based on `scikit-learn.base.BaseEstimator._get_param_names`
        # Copyright Gael Varoquaux, BSD 3 clause
        signature = inspect.signature(cls)
        parameters = [
            p
            for p in signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "Oh no."
                )

        return sorted([p.name for p in parameters])

    def _saveModelFilesBackup(self, backupName = "modelBackup"):
        oldBasePath = str(self.file_system.files.DIRECTORY)
        newBasePath = f"{oldBasePath}/{backupName}"
        try:
            shutil.rmtree(newBasePath)
        except:
            pass
        os.mkdir(newBasePath)
        shutil.move(f'{oldBasePath}/train', newBasePath)
        shutil.move(f'{oldBasePath}/test', newBasePath)
        try:
            shutil.move(f'{oldBasePath}/train_output.txt', newBasePath)
        except:
            pass
        try:
            shutil.move(f'{oldBasePath}/test_output.txt', newBasePath)
        except:
            pass

        os.makedirs(f'{oldBasePath}/train', exist_ok = True)
        os.makedirs(f'{oldBasePath}/test', exist_ok = True)

    def _cleanModelFiles(self):
        oldBasePath = str(self.file_system.files.DIRECTORY)
        shutil.rmtree(f'{oldBasePath}/train')
        shutil.rmtree(f'{oldBasePath}/test')
        try:
            os.remove(f'{oldBasePath}/train_output.txt')
        except:
            pass
        try:
            os.remove(f'{oldBasePath}/test_output.txt')
        except:
            pass

        os.makedirs(f'{oldBasePath}/train', exist_ok = True)
        os.makedirs(f'{oldBasePath}/test', exist_ok = True)

    def _restoreModelFromBackup(self, backupName = "modelBackup"):
        newBasePath = str(self.file_system.files.DIRECTORY)
        backupDirPath = f"{newBasePath}/{backupName}"
        try:
            shutil.rmtree(f"{newBasePath}/train")
            shutil.rmtree(f"{newBasePath}/test")
            os.remove(f"{newBasePath}/train_output.txt")
            os.remove(f"{newBasePath}/test_output.txt")
        except:
            pass
        shutil.move(f'{backupDirPath}/train', newBasePath)
        shutil.move(f'{backupDirPath}/test', newBasePath)
        shutil.move(f'{backupDirPath}/train_output.txt', newBasePath)
        shutil.move(f'{backupDirPath}/test_output.txt', newBasePath)

        shutil.rmtree(backupDirPath)

    def _check_params(self):
        """Check validity of parameters. Raise ValueError if errors are detected.

        If all parameters are valid, instantiate ``self.file_system`` by
        instantiating it with a :class:`srlearn.system_manager.FileSystem`
        """

        checks = (
            (
                self.n_estimators,
                (int,),
                (
                    lambda x: not isinstance(x, bool),
                    lambda x: x >= 1,
                ),
                "'n_estimators' must be an 'int' >= 1",
            ),
            (
                self.neg_pos_ratio,
                (int, float),
                (lambda x: not isinstance(x, bool), lambda x: x >= 1.0),
                "'neg_pos_ratio' must be 'int' or 'float'",
            ),
            (
                self.node_size,
                (int,),
                (
                    lambda x: not isinstance(x, bool),
                    lambda x: x >= 1,
                ),
                "'node_size' must be an 'int' >= 1",
            ),
            (
                self.max_tree_depth,
                (int,),
                (
                    lambda x: not isinstance(x, bool),
                    lambda x: x >= 1,
                ),
                "'max_tree_depth' must be an 'int' >= 1",
            ),
        )

        for param, types, constraints, message in checks:
            if not any([isinstance(param, t) for t in types]):
                raise ValueError(message)
            for c in constraints:
                if not c(param):
                    raise ValueError(message)

        # If all params are valid, allocate a FileSystem:
        self.file_system = FileSystem(path = self.path)

    # def to_json(self, file_name) -> None:
    #     """Serialize a learned model to json.

    #     Parameters
    #     ----------
    #     file_name : str (or pathlike)
    #         Path to a saved json file.

    #     Notes / Warnings
    #     ----------------

    #     Intended for locally saving/loading.

    #     .. warning::

    #         There could be major changes between releases, causing old model
    #         files to break."""
    #     check_is_fitted(self, "estimators_")

    #     with open(
    #         self.file_system.files.BRDNS_DIR.joinpath(
    #             "{0}.model".format(self.target)
    #         ),
    #         "r",
    #     ) as _fh:
    #         _model = _fh.read().splitlines()

    #     model_params = {
    #         **self.__dict__,
    #     }

    #     with open(file_name, "w") as _fh:
    #         _fh.write(
    #             json.dumps(
    #                 [
    #                     __version__,
    #                     _model,
    #                     self.estimators_,
    #                     model_params,
    #                     self._dotfiles,
    #                 ]
    #             )
    #         )

    # def from_json(self, file_name):
    #     """Load a learned model from json.

    #     Parameters
    #     ----------
    #     file_name : str (or pathlike)
    #         Path to a saved json file.

    #     Notes / Warnings
    #     ----------------

    #     Intended for locally saving/loading.

    #     .. warning::

    #         There could be major changes between releases, causing old model
    #         files to break. There are also *no checks* to ensure you are
    #         loading the correct object type.
    #     """

    #     with open(file_name, "r") as _fh:
    #         params = json.loads(_fh.read())

    #     if params[0] != __version__:
    #         logging.warning(
    #             "Version of loaded model ({0}) does not match srlearn version ({1}).".format(
    #                 params[0], __version__
    #             )
    #         )

    #     _model = params[1]
    #     _estimators = params[2]
    #     _model_parameters = params[3]

    #     try:
    #         self._dotfiles = params[4]
    #     except IndexError:
    #         self._dotfiles = None
    #         logging.warning(
    #             "Did not find dotfiles during load, srlearn.plotting may not work."
    #         )

    #     # 1. Loop over all class attributes of `BaseBoostedRelationalModel`
    #     #    except `background`, `node_size`, and `max_tree_depth`, which are
    #     #    handled by `Background` objects.
    #     # 2. Update an `_attributes` dictionary mapping attributes from JSON
    #     # 3. *If a key was not present in the JSON*: set it to the default value.
    #     # 4. Initialize self by unpacking the dictionary into arguments.
    #     _attributes = {
    #         "node_size": _model_parameters["node_size"],
    #         "max_tree_depth": _model_parameters["max_tree_depth"],
    #     }
    #     for key in set(BaseBoostedRelationalModel()._get_param_names()) - {"background", "node_size", "max_tree_depth"}:
    #         _attributes[key] = _model_parameters.get(
    #             key,
    #             BaseBoostedRelationalModel().__dict__[key],
    #         )
    #     self.__init__(**_attributes)

    #     self.estimators_ = _estimators

    #     # Currently allocates the File System.
    #     self._check_params()

    #     self.file_system.files.TREES_DIR.mkdir(parents=True)

    #     with open(
    #         self.file_system.files.BRDNS_DIR.joinpath(
    #             "{0}.model".format(self.target)
    #         ),
    #         "w",
    #     ) as _fh:
    #         _fh.write("\n".join(_model))

    #     for i, _tree in enumerate(_estimators):
    #         with open(
    #             self.file_system.files.TREES_DIR.joinpath(
    #                 "{0}Tree{1}.tree".format(self.target, i)
    #             ),
    #             "w",
    #         ) as _fh:
    #             _fh.write(_tree)

    #     return self

    # @property
    # def feature_importances_(self):
    #     """
    #     Return the features contained in a tree.

    #     Parameters
    #     ----------

    #     tree_number: int
    #         Index of the tree to read.
    #     """
    #     check_is_fitted(self, "estimators_")

    #     features = []

    #     for tree_number in range(self.n_estimators):
    #         _rules_string = self.estimators_[tree_number]
    #         features += parse_tree(
    #             _rules_string, (not self.background.use_std_logic_variables)
    #         )
    #     return Counter(features)

    def _get_dotfiles(self):
        dotFilePaths = glob(f"{self.file_system.files.DOT_DIR}/*.dot")
        dotfiles = []
        for filepath in dotFilePaths:
            with open(filepath) as _fh:
                dotfiles.append(_fh.read())
        self._dotfiles = dotfiles

    def _generate_dotimages(self):
        dotFilePaths = glob(f"{self.file_system.files.DOT_DIR}/*.dot")
        dotimages = []
        for filepath in dotFilePaths:
            dotimages.append(Source.from_file(filepath))
        self._dotimages = dotimages

    def _check_initialized(self):
        """Check for the estimator(s), raise an error if not found."""
        check_is_fitted(self, "estimators_")

    @staticmethod
    def _call_shell_command(shell_command, ignoreSTDOUT = True):
        """Start a new process to execute a shell command.

        This is intended for use in calling jar files. It opens a new process and
        waits for it to return 0.

        Parameters
        ----------
        shell_command : str
            A string representing a shell command.

        Returns
        -------
        None
        """

        _pid = subprocess.Popen(
            shell_command, 
            shell=True, 
            stdout = subprocess.DEVNULL if ignoreSTDOUT else None,
            stderr = subprocess.DEVNULL if ignoreSTDOUT else None
        )
        _status = _pid.wait()
        if _status != 0:
            raise RuntimeError(
                "Error when running shell command: {0}".format(shell_command)
            )

    def fit(self, database, ignoreSTDOUT = True):
        raise NotImplementedError

    def predict(self, database, ignoreSTDOUT = True):
        raise NotImplementedError

    def predict_proba(self, database, ignoreSTDOUT = True):
        raise NotImplementedError

    def __repr__(self):
        params = self._get_param_names()
        params.remove("max_tree_depth")
        params.remove("node_size")
        params = ", ".join([str(param) + "=" + repr(self.__dict__[param]) for param in params])
        return (
            self.__class__.__name__
            + f"({params})"
        )
