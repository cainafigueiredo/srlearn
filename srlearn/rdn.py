# Copyright © 2017, 2018, 2019 Alexander L. Hayes

"""
Relational Dependency Networks
"""

# TODO: Check and update class and method descriptions
# TODO: Use TreeBoostler as a dependent package
# TODO: Extract learning and inference time for all models
# TODO: Find a way to extract the mapping that TreeBoostler applies to the source model

import os
import re
import sys
import copy
import shutil
import pathlib
import warnings
import numpy as np

from .base import BaseBoostedRelationalModel
from .database import TransferLearningDatabase, Database

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from typing import Union
from glob import glob

# TODO: Install TreeBoostler as a dependency and change how it is imported.
PROJECT_DIR = os.path.dirname(__file__)
RELATED_WORK_PATH = os.path.abspath(f"{PROJECT_DIR}/../../../relatedWork")
sys.path.append(RELATED_WORK_PATH)
from TreeBoostler.revision import revision
from TreeBoostler.tboostsrl import tboostsrl
from TreeBoostler.transfer import transfer

from .background import Background

warnings.simplefilter("default")

class RDNBoost(BaseBoostedRelationalModel):
    def __init__(
        self, 
        *, 
        n_estimators = 10, 
        node_size = 2, 
        max_tree_depth = 3, 
        neg_pos_ratio = 2,
        number_of_clauses: int = 8,
        number_of_cycles: int = 100,
        path = None,
    ):
        super().__init__(
            n_estimators = n_estimators, 
            node_size = node_size, 
            max_tree_depth = max_tree_depth, 
            neg_pos_ratio = neg_pos_ratio, 
            number_of_clauses = number_of_clauses,
            number_of_cycles = number_of_cycles,
            solver = "BoostSRL",
            path = path
        )

    def _check_params(self):
        super()._check_params()

    def _callModel(
        self,
        database: Database,
        trainOrTest = "train",
        ignoreSTDOUT = True
    ):
        _jar = str(self.file_system.files.BOOSTSRL_BACKEND)

        modes = database.modes
        target = database.getTargetRelation()

        background = Background(
            modes = modes,
            number_of_clauses = self.number_of_clauses,
            number_of_cycles = self.number_of_cycles,
            node_size = self.node_size,
            max_tree_depth = self.max_tree_depth
        )

        if trainOrTest == "train":
            background.write(
                filename="train", location=self.file_system.files.TRAIN_DIR
            )

            database.write(
                filename="train", location=self.file_system.files.TRAIN_DIR
            )

            _CALL = (
                "java -jar "
                + _jar
                + " -l -train "
                + str(self.file_system.files.TRAIN_DIR)
                + " -target "
                + target
                + " -trees "
                + str(self.n_estimators)
                + " -negPosRatio "
                + str(self.neg_pos_ratio)
                + " 2>&1 | tee "
                + str(self.file_system.files.TRAIN_LOG)
            )
        
        else:
            background.write(
                filename="test", location=self.file_system.files.TEST_DIR
            )

            database.write(
                filename="test", location=self.file_system.files.TEST_DIR
            )

            _CALL = (
                "java -jar "
                + _jar
                + " -i -test "
                + str(self.file_system.files.TEST_DIR)
                + " -model "
                + str(self.file_system.files.MODELS_DIR)
                + " -target "
                + target
                + " -trees "
                + str(self.n_estimators)
                + " -aucJarPath "
                + str(self.file_system.files.AUC_JAR)
                + " 2>&1 | tee "
                + str(self.file_system.files.TEST_LOG)
            )

        self._call_shell_command(_CALL, ignoreSTDOUT = ignoreSTDOUT)

    def fit(self, database, ignoreSTDOUT = True):
        self._check_params()

        if not isinstance(database, Database):
            raise ValueError("Database should be an instance of Database class.")

        target = database.getTargetRelation()

        self._callModel(database, trainOrTest = "train", ignoreSTDOUT = ignoreSTDOUT)

        # Read the trees from files.
        _estimators = []
        for _tree_number in range(self.n_estimators):
            with open(
                self.file_system.files.TREES_DIR.joinpath(
                    "{0}Tree{1}.tree".format(target, _tree_number)
                )
            ) as _fh:
                _estimators.append(_fh.read())

        self._get_dotfiles()
        self.estimators_ = _estimators

        # TODO: Get training time. It can be extracted from the training output file. Where can we store this metric? Maybe we can store in `self._prediction_metrics`. In that case, is it better to change the name of this attribute to something like `_metrics`?

        return self

    def _run_inference(self, database, ignoreSTDOUT = True) -> dict:
        """Run inference mode on the BoostSRL Jar files.

        This is a helper method for ``self.predict`` and ``self.predict_proba``
        """

        self._check_initialized()

        self._callModel(
            database,
            trainOrTest = "test",
            ignoreSTDOUT = ignoreSTDOUT
        )

        # Read the threshold
        with open(self.file_system.files.TEST_LOG, "r") as _fh:
            log = _fh.read()
            self._prediction_metrics = {}
            self._prediction_metrics["threshold"] = re.findall(r".*threshold = (\d*\.?\d*)", log)[0]
            self._prediction_metrics["cll"] = re.findall(r".*CLL.*= (-?\d*\.?\d*)", log)[0]
            self._prediction_metrics["aucROC"] = re.findall(r".*AUC ROC.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["aucPR"] = re.findall(r".*AUC PR.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["prec"] = re.findall(r".*Precision.*= (\d*\.?\d*).*", log)[0]
            self._prediction_metrics["rec"] = re.findall(r".*Recall.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["f1"] = re.findall(r"\%.*F1.*= (\d*\.?\d*)", log)[0]

            self._prediction_metrics["threshold"] = self._prediction_metrics["threshold"] if self._prediction_metrics["threshold"] != "" else "nan"
            self._prediction_metrics["cll"] = self._prediction_metrics["cll"] if self._prediction_metrics["cll"] != "" else "nan"
            self._prediction_metrics["aucROC"] = self._prediction_metrics["aucROC"] if self._prediction_metrics["aucROC"] != "" else "nan"
            self._prediction_metrics["aucPR"] = self._prediction_metrics["aucPR"] if self._prediction_metrics["aucPR"] != "" else "nan"
            self._prediction_metrics["prec"] = self._prediction_metrics["prec"] if self._prediction_metrics["prec"] != "" else "nan"
            self._prediction_metrics["rec"] = self._prediction_metrics["rec"] if self._prediction_metrics["rec"] != "" else "nan"
            self._prediction_metrics["f1"] = self._prediction_metrics["f1"] if self._prediction_metrics["f1"] != "" else "nan"
            # TODO: Get inference time. It can be extracted from the same file from which the metrics above were extracted.

    def predict(self, database, threshold = None, ignoreSTDOUT = True):
        """Use the learned model to predict on new data.

        Parameters
        ----------
        database : :class:`srlearn.Database`
            Database containing examples and facts.
        threshold: float (default: None)
            Classification threshold. If None, it uses the threshold specified by self._threshold.
            
        Returns
        -------
        results : ndarray
            Positive or negative class.
        """

        predictedProbabilities = self.predict_proba(database, ignoreSTDOUT = ignoreSTDOUT)

        threshold = threshold if threshold is not None else self._threshold

        predictedLabels = np.greater(
            predictedProbabilities, threshold
        )

        if threshold != self._prediction_metrics["threshold"]:
            trueLabels = self.classes_
            self._prediction_metrics["acc"] = accuracy_score(trueLabels, predictedLabels)
            self._prediction_metrics["pr"] = precision_score(trueLabels, predictedLabels)
            self._prediction_metrics["rec"] = recall_score(trueLabels, predictedLabels)
            self._prediction_metrics["f1"] = f1_score(trueLabels, predictedLabels)

            self._prediction_metrics["acc"] = self._prediction_metrics["acc"] if self._prediction_metrics["acc"] != "" else "nan"
            self._prediction_metrics["prec"] = self._prediction_metrics["prec"] if self._prediction_metrics["prec"] != "" else "nan"
            self._prediction_metrics["rec"] = self._prediction_metrics["rec"] if self._prediction_metrics["rec"] != "" else "nan"
            self._prediction_metrics["f1"] = self._prediction_metrics["f1"] if self._prediction_metrics["f1"] != "" else "nan"

        return predictedLabels

    def predict_proba(self, database, ignoreSTDOUT = True):
        """Return class probabilities.

        Parameters
        ----------
        database : :class:`srlearn.Database`
            Database containing examples and facts.

        Returns
        -------
        results : ndarray
            Probability of belonging to the positive class
        """

        self._run_inference(database, ignoreSTDOUT = ignoreSTDOUT)

        target = database.getTargetRelation()

        _results_db = self.file_system.files.TEST_DIR.joinpath(
            "results_" + target + ".db"
        )
        _classes, _results = np.genfromtxt(
            _results_db,
            delimiter=") ",
            usecols=(0, 1),
            converters={0: lambda s: 0 if s[0] == 33 else 1},
            unpack=True,
        )

        _neg = _results[_classes == 0]
        _pos = _results[_classes == 1]
        _results2 = np.concatenate((_pos, 1 - _neg), axis=0)

        self.classes_ = _classes

        return _results2

class RDNBoostTransferLearning(BaseBoostedRelationalModel):
    def __init__(
        self,
        *,
        n_estimators = 10,
        node_size = 2,
        max_tree_depth = 3,
        neg_pos_ratio = 2,
        number_of_clauses: int = 8,
        number_of_cycles: int = 100,
        source_utility_alpha = 1,
        target_utility_alpha = 1,
        utility_alpha_set_iter = 1,
        path = None
    ):
        """Initialize a BoostedRDN Transfer Learner

        Parameters
        ----------
        n_estimators : int, optional (default: 10)
            Number of trees to fit
        node_size : int, optional (default: 2)
            Maximum number of literals in each node.
        max_tree_depth : int, optional (default: 3)
            Maximum number of nodes from root to leaf (height) in the tree.
        neg_pos_ratio : int or float, optional (default: 2)
            Ratio of negative to positive examples used during learning.
        source_utility_alpha : float, optional (default: 1)
            Source alpha hyperparameter for alpha-fairness utility function.
        target_utility_alpha : float, optional (default: 1)
            Target alpha hyperparameter for alpha-fairness utility function.
        utility_alpha_set_iter : int, optional (default: 1)
            Learning iteration where source utility alpha and target utility alpha will be set to `source_utility_alpha` and `target_utility_alpha`, respectively. Before this iteration, both of them will be set to 1.
            
        Attributes
        ----------
        estimators_ : array, shape (n_estimators)
            Return the boosted regression trees
        feature_importances_ : array, shape (n_features)
            Return the feature importances (based on how often each feature appears)
        """

        super().__init__(
            n_estimators = n_estimators,
            node_size = node_size,
            max_tree_depth = max_tree_depth,
            neg_pos_ratio = neg_pos_ratio,
            number_of_clauses = number_of_clauses,
            number_of_cycles = number_of_cycles,
            solver = "BoostSRLTransferLearner",
            path = path
        )
        self.source_utility_alpha = source_utility_alpha
        self.target_utility_alpha = target_utility_alpha
        self.utility_alpha_set_iter = utility_alpha_set_iter

    def _check_params(self):
        if not (self.source_utility_alpha >= 0):
            raise ValueError("Source alpha for alpha-fairness should be greater than or equal to zero.")
        if not (self.target_utility_alpha >= 0):
            raise ValueError("Target alpha for alpha-fairness should be greater than or equal to zero.")
        super()._check_params()

    def _callModel(
        self,
        database: Union[Database,TransferLearningDatabase],
        trainOrTest = "train",
        ignoreSTDOUT = True
    ):
        _jar = str(self.file_system.files.BOOSTSRL_TRANFER_LEARNER_BACKEND)

        modes = database.modes
        target = database.getTargetRelation()

        background = Background(
            modes = modes,
            number_of_clauses = self.number_of_clauses,
            number_of_cycles = self.number_of_cycles,
            node_size = self.node_size,
            max_tree_depth = self.max_tree_depth
        )

        if trainOrTest == "train":
            background.write(
                filename="train", location=self.file_system.files.TRAIN_DIR
            )

            database.write(
                filename="train", location=self.file_system.files.TRAIN_DIR
            )

            _CALL = (
                "java -jar "
                + _jar
                + " -l -train "
                + str(self.file_system.files.TRAIN_DIR)
                + " -target "
                + target
                + " -trees "
                + str(self.n_estimators)
                + " -negPosRatio "
                + str(self.neg_pos_ratio)
                + " -sourceUtilityAlpha "
                + str(self.source_utility_alpha)
                + " -targetUtilityAlpha "
                + str(self.target_utility_alpha)
                + " -utilityAlphaSetIter "
                + str(self.utility_alpha_set_iter)
                + " 2>&1 | tee "
                + str(self.file_system.files.TRAIN_LOG)
            )
        
        else:
            background.write(
                filename="test", location=self.file_system.files.TEST_DIR
            )

            database.write(
                filename="test", location=self.file_system.files.TEST_DIR
            )

            _CALL = (
                "java -jar "
                + _jar
                + " -i -test "
                + str(self.file_system.files.TEST_DIR)
                + " -model "
                + str(self.file_system.files.MODELS_DIR)
                + " -target "
                + target
                + " -trees "
                + str(self.n_estimators)
                + " -aucJarPath "
                + str(self.file_system.files.AUC_JAR)
                + " 2>&1 | tee "
                + str(self.file_system.files.TEST_LOG)
            )

        self._call_shell_command(_CALL, ignoreSTDOUT = ignoreSTDOUT)

    def fit(self, database: TransferLearningDatabase, ignoreSTDOUT = True):
        self._check_params()

        if not isinstance(database, TransferLearningDatabase):
            raise ValueError("Database should be an instance of TransferLearningDatabase class.")

        target = database.getTargetRelation()

        self._callModel(
            database,
            trainOrTest = "train",
            ignoreSTDOUT = ignoreSTDOUT
        )

        # Read the trees from files.
        _estimators = []
        for _tree_number in range(self.n_estimators):
            with open(
                self.file_system.files.TREES_DIR.joinpath(
                    "{0}Tree{1}.tree".format(target, _tree_number)
                )
            ) as _fh:
                _estimators.append(_fh.read())

        self._get_dotfiles()
        self.estimators_ = _estimators

        # TODO: Get training time. It can be extracted from the training output file. Where can we store this metric? Maybe we can store in `self._prediction_metrics`. In that case, is it better to change the name of this attribute to something like `_metrics`?

        return self

    def _run_inference(self, database: Database, ignoreSTDOUT = True) -> dict:
        """Run inference mode on the BoostSRL Jar files.

        This is a helper method for ``self.predict`` and ``self.predict_proba``
        """

        self._check_initialized()

        self._callModel(
            database, 
            trainOrTest = "test",
            ignoreSTDOUT = ignoreSTDOUT 
        )

        # Read the threshold
        with open(self.file_system.files.TEST_LOG, "r") as _fh:
            log = _fh.read()
            self._prediction_metrics = {}
            self._prediction_metrics["threshold"] = re.findall(r".*threshold = (\d*\.?\d*)", log)[0]
            self._prediction_metrics["cll"] = re.findall(r".*CLL.*= (-?\d*\.?\d*)", log)[0]
            self._prediction_metrics["aucROC"] = re.findall(r".*AUC ROC.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["aucPR"] = re.findall(r".*AUC PR.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["prec"] = re.findall(r".*Precision.*= (\d*\.?\d*).*", log)[0]
            self._prediction_metrics["rec"] = re.findall(r".*Recall.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["f1"] = re.findall(r"\%.*F1.*= (\d*\.?\d*)", log)[0]

            self._prediction_metrics["threshold"] = self._prediction_metrics["threshold"] if self._prediction_metrics["threshold"] != "" else "nan"
            self._prediction_metrics["cll"] = self._prediction_metrics["cll"] if self._prediction_metrics["cll"] != "" else "nan"
            self._prediction_metrics["aucROC"] = self._prediction_metrics["aucROC"] if self._prediction_metrics["aucROC"] != "" else "nan"
            self._prediction_metrics["aucPR"] = self._prediction_metrics["aucPR"] if self._prediction_metrics["aucPR"] != "" else "nan"
            self._prediction_metrics["prec"] = self._prediction_metrics["prec"] if self._prediction_metrics["prec"] != "" else "nan"
            self._prediction_metrics["rec"] = self._prediction_metrics["rec"] if self._prediction_metrics["rec"] != "" else "nan"
            self._prediction_metrics["f1"] = self._prediction_metrics["f1"] if self._prediction_metrics["f1"] != "" else "nan"
            # TODO: Get inference time. It can be extracted from the same file from which the metrics above were extracted.
        
    def predict(self, database, threshold = None, ignoreSTDOUT = True):
        """Use the learned model to predict on new data.

        Parameters
        ----------
        database : :class:`srlearn.Database`
            Database containing examples and facts.
        threshold: float (default: None)
            Classification threshold. If None, it uses the threshold specified by self._threshold.
            
        Returns
        -------
        results : ndarray
            Positive or negative class.
        """

        predictedProbabilities = self.predict_proba(database, ignoreSTDOUT = ignoreSTDOUT)

        threshold = threshold if threshold is not None else self._threshold

        predictedLabels = np.greater(
            predictedProbabilities, threshold
        )

        if threshold != self._prediction_metrics["threshold"]:
            trueLabels = self.classes_
            self._prediction_metrics["acc"] = accuracy_score(trueLabels, predictedLabels)
            self._prediction_metrics["pr"] = precision_score(trueLabels, predictedLabels)
            self._prediction_metrics["rec"] = recall_score(trueLabels, predictedLabels)
            self._prediction_metrics["f1"] = f1_score(trueLabels, predictedLabels)

            self._prediction_metrics["acc"] = self._prediction_metrics["acc"] if self._prediction_metrics["acc"] != "" else "nan"
            self._prediction_metrics["prec"] = self._prediction_metrics["prec"] if self._prediction_metrics["prec"] != "" else "nan"
            self._prediction_metrics["rec"] = self._prediction_metrics["rec"] if self._prediction_metrics["rec"] != "" else "nan"
            self._prediction_metrics["f1"] = self._prediction_metrics["f1"] if self._prediction_metrics["f1"] != "" else "nan"

        return predictedLabels

    def predict_proba(self, database, ignoreSTDOUT = True):
        """Return class probabilities.

        Parameters
        ----------
        database : :class:`srlearn.Database`
            Database containing examples and facts.

        Returns
        -------
        results : ndarray
            Probability of belonging to the positive class
        """

        self._run_inference(database, ignoreSTDOUT = ignoreSTDOUT)
        target = database.getTargetRelation()

        _results_db = self.file_system.files.TEST_DIR.joinpath(
            "results_" + target + ".db"
        )
        _classes, _results = np.genfromtxt(
            _results_db,
            delimiter=") ",
            usecols=(0, 1),
            converters={0: lambda s: 0 if s[0] == 33 else 1},
            unpack=True,
        )

        _neg = _results[_classes == 0]
        _pos = _results[_classes == 1]
        _results2 = np.concatenate((_pos, 1 - _neg), axis=0)

        self.classes_ = _classes

        return _results2

class TreeBoostler(BaseBoostedRelationalModel):
    def __init__(
        self,
        *,
        searchArgPermutation: bool = True,
        allowSameTargetMap: bool = False,
        refine: bool = True,
        maxRevisionIterations: int = 1,
        n_estimators: int = 10,
        node_size: int = 2,
        max_tree_depth: int = 3,
        neg_pos_ratio: int = 2,
        number_of_clauses: int = 8,
        number_of_cycles: int = 100,
        path = None
    ):
        # TODO: Add a description to the class.

        super().__init__(
            n_estimators = n_estimators,
            node_size = node_size,
            max_tree_depth = max_tree_depth,
            neg_pos_ratio = neg_pos_ratio,
            number_of_clauses = number_of_clauses,
            number_of_cycles = number_of_cycles,
            solver = "TreeBoostler",
            path = path
        )
        self.searchArgPermutation = searchArgPermutation
        self.allowSameTargetMap = allowSameTargetMap
        self.refine = refine
        self.maxRevisionIterations = maxRevisionIterations

    def _check_params(self):
        super()._check_params()

    def _callModel(
        self,
        database: Database,
        transferFile = None,
        refine = None,
        trainOrTest = "train",
        ignoreSTDOUT = True
    ):
        _jar = str(self.file_system.files.TREEBOOSTLER_BACKEND)

        modes = database.modes
        target = database.getTargetRelation()

        background = Background(
            modes = modes,
            number_of_clauses = self.number_of_clauses,
            number_of_cycles = self.number_of_cycles,
            node_size = self.node_size,
            max_tree_depth = self.max_tree_depth
        )

        if trainOrTest == "train":
            background.write(
                filename="train", location=self.file_system.files.TRAIN_DIR
            )

            database.write(
                filename="train", location=self.file_system.files.TRAIN_DIR
            )

            if refine:
                with open(os.path.join(str(self.file_system.files.DIRECTORY), "refine.txt"), "w") as f:
                    f.write("\n".join(refine))
                    f.write("\n")

            if transferFile:
                with open(os.path.join(str(self.file_system.files.DIRECTORY), "transfer.txt"), "w") as f:
                    f.write("\n".join(transferFile))
                    f.write("\n")

            _CALL = (
                "java -jar "
                + _jar
                + " -l -train "
                + str(self.file_system.files.TRAIN_DIR)
                + (' -refine refine.txt ' if refine else '' )
                + (' -transfer transfer.txt ' if transferFile else '')
                + " -target "
                + target
                + " -trees "
                + str(self.n_estimators)
                + " -negPosRatio "
                + str(self.neg_pos_ratio)
                + " 2>&1 | tee "
                + str(self.file_system.files.TRAIN_LOG)
            )
        
        else:
            background.write(
                filename="test", location=self.file_system.files.TEST_DIR
            )

            database.write(
                filename="test", location=self.file_system.files.TEST_DIR
            )

            _CALL = (
                "java -jar "
                + _jar
                + " -i -test "
                + str(self.file_system.files.TEST_DIR)
                + " -model "
                + str(self.file_system.files.MODELS_DIR)
                + " -target "
                + target
                + " -trees "
                + str(self.n_estimators)
                + " -aucJarPath "
                + str(self.file_system.files.AUC_JAR)
                + " 2>&1 | tee "
                + str(self.file_system.files.TEST_LOG)
            )

        self._call_shell_command(_CALL, ignoreSTDOUT = ignoreSTDOUT)

    def _get_structured_tree(self, treenumber=1):
        '''Use the get_will_produced_tree function to get the WILL-Produced Tree #1
           and returns it as objects with nodes, std devs and number of examples reached.'''
        def get_results(groups):
            #std dev, neg, pos
            # std dev with comma, is this supposed to happen?
            ret = [float(groups[0].replace(',','.')), 0, 0]
            if len(groups) > 1:
                match = re.findall(r'\#pos=([\d.]*).*', groups[1])
                if match:
                    ret[2] = int(match[0].replace('.',''))
                match = re.findall(r'\#neg=([\d.]*)', groups[1])
                if match:
                    ret[1] = int(match[0].replace('.',''))
            return ret

        lines = self._get_will_produced_tree(treenumber=treenumber)
        current = []
        stack = []
        target = None
        nodes = {}
        leaves = {}

        for line in lines:
            if not target:
                match = re.match('\s*\%\s*FOR\s*(\w+\([\w,\s]*\)):', line)
                if match:
                    target = match.group(1)
            match = re.match('.*if\s*\(\s*([\w\(\),\s]*)\s*\).*', line)
            if match:
                nodes[','.join(current)] = match.group(1).strip()
                stack.append(current+['false'])
                current.append('true')
            match = re.match('.*[then|else] return .*;\s*\/\/\s*std dev\s*=\s*([\d,.\-e]*),.*\/\*\s*(.*)\s*\*\/.*', line)
            if match:
                leaves[','.join(current)] = get_results(match.groups()) #float(match.group(1))
                if len(stack):
                    current = stack.pop()
            else:
                match = re.match('.*[then|else] return .*;\s*\/\/\s*.*', line)
                if match:
                    leaves[','.join(current)] = get_results(['0'] + list(match.groups())) #float(match.group(1))
                    if len(stack):
                        current = stack.pop()
        return [target, nodes, leaves]

    def _get_will_produced_tree(self, treenumber = 1):
        '''Return the WILL-Produced Tree'''
        combine = 'Combined' if self.n_estimators > 1 and treenumber=='combine' else '#' + str(treenumber)
        willTreesFilePath = glob(f'{str(self.file_system.files.MODELS_DIR)}/WILLtheories/*_learnedWILLregressionTrees.txt')[0]
        with open(willTreesFilePath, 'r') as f:
            text = f.read()
        line = re.findall(r'%%%%%  WILL-Produced Tree '+ combine +' .* %%%%%[\s\S]*% Clauses:', text)
        splitline = (line[0].split('\n'))[2:]
        for i in range(len(splitline)):
            if splitline[i] == '% Clauses:':
                return splitline[:i-2]

    def _get_variances(self, treenumber = 1):
        '''Return variances of nodes'''
        with open(f'{str(self.file_system.files.TRAIN_DIR)}/train_learn_dribble.txt', 'r') as f:
            text = f.read()
        line = re.findall(r'% Path: '+ str(treenumber-1) + ';([\w,]*)\sComparing variance: ([\d.\w\-]*) .*\sComparing variance: ([\d.\w\-]*) .*', text)
        ret = {}
        for item in line:
            ret[item[0]] = [float(item[1]), float(item[2])]
        return ret

    def fit(
        self, 
        sourceDatabase: Database, 
        targetDatabase: Database, 
        ignoreSTDOUT = True
    ):
        self._check_params()

        if not isinstance(sourceDatabase, Database):
            raise ValueError("Source database should be an instance of Database class.")

        if not isinstance(targetDatabase, Database):
            raise ValueError("Target database should be an instance of Database class.")

        # Learning model from source domain
        sourceDomainTargetPredicate = sourceDatabase.getTargetRelation()

        sourceModes = sourceDatabase.modes

        self._callModel(
            sourceDatabase,
            transferFile = None, 
            refine = None,
            trainOrTest = "train",
            ignoreSTDOUT = ignoreSTDOUT
        )

        sourceStructure = []
        for i in range(self.n_estimators):
            sourceStructure.append(self._get_structured_tree(treenumber=i+1).copy())

        # Transfer source model and refining it
        targetDomainTargetPredicate = targetDatabase.getTargetRelation()

        targetModes = targetDatabase.modes
    
        transferFile = transfer.get_transfer_file(
            sourceModes, 
            targetModes, 
            sourceDomainTargetPredicate,
            targetDomainTargetPredicate,
            searchArgPermutation = self.searchArgPermutation,
            allowSameTargetMap = self.allowSameTargetMap
        )

        refine = revision.get_boosted_refine_file(sourceStructure) if self.refine else None

        self._saveModelFilesBackup("best")

        self._callModel(
            targetDatabase,
            transferFile = transferFile, 
            refine = refine,
            trainOrTest = "train",
            ignoreSTDOUT = ignoreSTDOUT
        )

        self._run_inference(
            targetDatabase,
            ignoreSTDOUT = ignoreSTDOUT
        )
        trainPredictionMetrics = copy.deepcopy(self._prediction_metrics)

        targetStructure = []
        for i in range(self.n_estimators):
            targetStructure.append(self._get_structured_tree(treenumber=i+1).copy())

        bestStructure = copy.deepcopy(targetStructure)
        bestResults = copy.deepcopy(trainPredictionMetrics)
        bestCLL = float(bestResults['cll'])
        variances = [self._get_variances(treenumber=i+1) for i in range(self.n_estimators)]

        self._saveModelFilesBackup("best")

        if self.refine:
            for revisionIter in range(self.maxRevisionIterations):
                found_better = False
                candidate = revision.get_boosted_candidate(bestStructure, variances)
                if not len(candidate):
                    candidate = revision.get_boosted_candidate(bestStructure, variances, no_pruning=True)

                self._callModel(
                    targetDatabase,
                    transferFile = None, 
                    refine = candidate,
                    ignoreSTDOUT = ignoreSTDOUT
                )

                self._run_inference(
                    targetDatabase,
                    ignoreSTDOUT = ignoreSTDOUT
                )
                predictionMetrics = copy.deepcopy(self._prediction_metrics)
                curCLL = float(predictionMetrics["cll"])

                if curCLL > bestCLL:
                    found_better = True
                    bestCLL = float(predictionMetrics['cll'])
                    bestStructure = []
                    for i in range(self.n_estimators):
                        bestStructure.append(self._get_structured_tree(treenumber=i+1).copy())
                    bestResults = copy.deepcopy(predictionMetrics)
                    self._saveModelFilesBackup("best")

                else: 
                    self._cleanModelFiles()

                if found_better == False:
                    break

        # Read the trees from files.
        _estimators = []
        for _tree_number in range(self.n_estimators):
            with open(
                os.path.join(
                    str(self.file_system.files.DIRECTORY), 
                    "best/train/models/bRDNs/Trees",
                    "{0}Tree{1}.tree".format(targetDomainTargetPredicate, _tree_number)
                )
            ) as _fh:
                _estimators.append(_fh.read())

        self._get_dotfiles()
        self.estimators_ = _estimators

        del self._prediction_metrics

        self._restoreModelFromBackup("best")

        return self

    def _run_inference(
        self, 
        database: Database,
        ignoreSTDOUT = True
    ) -> dict:
        """Run inference mode on the BoostSRL Jar files.

        This is a helper method for ``self.predict`` and ``self.predict_proba``
        """

        self._callModel(
            database,
            transferFile = None, 
            refine = None,
            trainOrTest = "test",
            ignoreSTDOUT = ignoreSTDOUT
        )

        # Read the threshold
        with open(self.file_system.files.TEST_LOG, "r") as _fh:
            log = _fh.read()
            self._prediction_metrics = {}
            self._prediction_metrics["threshold"] = re.findall(r".*threshold = (\d*\.?\d*)", log)[0]
            self._prediction_metrics["cll"] = re.findall(r".*CLL.*= (-?\d*\.?\d*)", log)[0]
            self._prediction_metrics["aucROC"] = re.findall(r".*AUC ROC.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["aucPR"] = re.findall(r".*AUC PR.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["prec"] = re.findall(r".*Precision.*= (\d*\.?\d*).*", log)[0]
            self._prediction_metrics["rec"] = re.findall(r".*Recall.*= (\d*\.?\d*)", log)[0]
            self._prediction_metrics["f1"] = re.findall(r"\%.*F1.*= (\d*\.?\d*)", log)[0]

            self._prediction_metrics["threshold"] = self._prediction_metrics["threshold"] if self._prediction_metrics["threshold"] != "" else "nan"
            self._prediction_metrics["cll"] = self._prediction_metrics["cll"] if self._prediction_metrics["cll"] != "" else "nan"
            self._prediction_metrics["aucROC"] = self._prediction_metrics["aucROC"] if self._prediction_metrics["aucROC"] != "" else "nan"
            self._prediction_metrics["aucPR"] = self._prediction_metrics["aucPR"] if self._prediction_metrics["aucPR"] != "" else "nan"
            self._prediction_metrics["prec"] = self._prediction_metrics["prec"] if self._prediction_metrics["prec"] != "" else "nan"
            self._prediction_metrics["rec"] = self._prediction_metrics["rec"] if self._prediction_metrics["rec"] != "" else "nan"
            self._prediction_metrics["f1"] = self._prediction_metrics["f1"] if self._prediction_metrics["f1"] != "" else "nan"
            # TODO: Get inference time. It can be extracted from the same file from which the metrics above were extracted.

    def predict(self, database, threshold = None, ignoreSTDOUT = True):
        """Use the learned model to predict on new data.

        Parameters
        ----------
        database : :class:`srlearn.Database`
            Database containing examples and facts.
        threshold: float (default: None)
            Classification threshold. If None, it uses the threshold specified by self._threshold.
            
        Returns
        -------
        results : ndarray
            Positive or negative class.
        """

        predictedProbabilities = self.predict_proba(database, ignoreSTDOUT = ignoreSTDOUT)

        threshold = threshold if threshold is not None else self._threshold

        predictedLabels = np.greater(
            predictedProbabilities, threshold
        )

        if threshold != self._prediction_metrics["threshold"]:
            trueLabels = self.classes_
            self._prediction_metrics["acc"] = accuracy_score(trueLabels, predictedLabels)
            self._prediction_metrics["pr"] = precision_score(trueLabels, predictedLabels)
            self._prediction_metrics["rec"] = recall_score(trueLabels, predictedLabels)
            self._prediction_metrics["f1"] = f1_score(trueLabels, predictedLabels)

            self._prediction_metrics["acc"] = self._prediction_metrics["acc"] if self._prediction_metrics["acc"] != "" else "nan"
            self._prediction_metrics["prec"] = self._prediction_metrics["prec"] if self._prediction_metrics["prec"] != "" else "nan"
            self._prediction_metrics["rec"] = self._prediction_metrics["rec"] if self._prediction_metrics["rec"] != "" else "nan"
            self._prediction_metrics["f1"] = self._prediction_metrics["f1"] if self._prediction_metrics["f1"] != "" else "nan"

        return predictedLabels

    def predict_proba(self, database, ignoreSTDOUT = True):
        """Return class probabilities.

        Parameters
        ----------
        database : :class:`srlearn.Database`
            Database containing examples and facts.

        Returns
        -------
        results : ndarray
            Probability of belonging to the positive class
        """

        self._run_inference(database, ignoreSTDOUT = ignoreSTDOUT)

        target = database.getTargetRelation()

        _results_db = self.file_system.files.TEST_DIR.joinpath(
            "results_" + target + ".db"
        )
        _classes, _results = np.genfromtxt(
            _results_db,
            delimiter=") ",
            usecols=(0, 1),
            converters={0: lambda s: 0 if s[0] == 33 else 1},
            unpack=True,
        )

        _neg = _results[_classes == 0]
        _pos = _results[_classes == 1]
        _results2 = np.concatenate((_pos, 1 - _neg), axis=0)

        self.classes_ = _classes

        return _results2