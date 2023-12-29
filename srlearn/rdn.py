# Copyright © 2017, 2018, 2019 Alexander L. Hayes

"""
Relational Dependency Networks
"""

import os
import re
import sys
import pathlib
import warnings
import numpy as np

from .base import BaseBoostedRelationalModel
from .database import TransferLearningDatabase, Database

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

warnings.simplefilter("default")

class RDNBoost(BaseBoostedRelationalModel):
    def __init__(
        self, 
        *, 
        background = None, 
        target = "None", 
        n_estimators = 10, 
        node_size = 2, 
        max_tree_depth = 3, 
        neg_pos_ratio = 2,
        path = None,
    ):
        super().__init__(
            background = background, 
            target = target, 
            n_estimators = n_estimators, 
            node_size = node_size, 
            max_tree_depth = max_tree_depth, 
            neg_pos_ratio = neg_pos_ratio, 
            solver = "BoostSRL",
            path = path
        )

    def _check_params(self):
        super()._check_params()

    def fit(self, database, ignoreSTDOUT = True):
        self._check_params()

        if not isinstance(database, Database):
            raise ValueError("Database should be an instance of Database class.")

        # Write the background to file.
        self.background.write(
            filename="train", location=self.file_system.files.TRAIN_DIR
        )

        # Write the data to files.
        database.write(
            filename="train", location=self.file_system.files.TRAIN_DIR
        )

        _jar = str(self.file_system.files.BOOSTSRL_BACKEND)

        _CALL = (
            "java -jar "
            + _jar
            + " -l -train "
            + str(self.file_system.files.TRAIN_DIR)
            + " -target "
            + self.target
            + " -trees "
            + str(self.n_estimators)
            + " -negPosRatio "
            + str(self.neg_pos_ratio)
            + " 2>&1 | tee "
            + str(self.file_system.files.TRAIN_LOG)
        )

        # Call the constructed command.
        self._call_shell_command(_CALL, ignoreSTDOUT = ignoreSTDOUT)

        # Read the trees from files.
        _estimators = []
        for _tree_number in range(self.n_estimators):
            with open(
                self.file_system.files.TREES_DIR.joinpath(
                    "{0}Tree{1}.tree".format(self.target, _tree_number)
                )
            ) as _fh:
                _estimators.append(_fh.read())

        self._get_dotfiles()
        self.estimators_ = _estimators

        # TODO: Get training time. It can be extracted from the training output file. Where can we store this metric? Maybe we can store in `self._prediction_metrics`. In that case, is it better to change the name of this attribute to something like `_metrics`?

        return self

    def _run_inference(self, database, ignoreSTDOUT = True) -> None:
        """Run inference mode on the BoostSRL Jar files.

        This is a helper method for ``self.predict`` and ``self.predict_proba``
        """

        self._check_initialized()

        # Write the background to file.
        self.background.write(
            filename="test", location=self.file_system.files.TEST_DIR
        )

        # Write the data to files.
        database.write(filename="test", location=self.file_system.files.TEST_DIR)
       
        _jar = str(self.file_system.files.BOOSTSRL_BACKEND)

        _CALL = (
            "java -jar "
            + _jar
            + " -i -test "
            + str(self.file_system.files.TEST_DIR)
            + " -model "
            + str(self.file_system.files.MODELS_DIR)
            + " -target "
            + self.target
            + " -trees "
            + str(self.n_estimators)
            + " -aucJarPath "
            + str(self.file_system.files.AUC_JAR)
            + " 2>&1 | tee "
            + str(self.file_system.files.TEST_LOG)
        )

        self._call_shell_command(_CALL, ignoreSTDOUT = ignoreSTDOUT)

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

        _results_db = self.file_system.files.TEST_DIR.joinpath(
            "results_" + self.target + ".db"
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
        background = None,
        target = "None",
        n_estimators = 10,
        node_size = 2,
        max_tree_depth = 3,
        neg_pos_ratio = 2,
        utility_alpha = 1,
        path = None
    ):
        """Initialize a BoostedRDN Transfer Learner

        Parameters
        ----------
        background : :class:`srlearn.background.Background` (default: None)
            Background knowledge with respect to the database
        target : str (default: "None")
            Target predicate to learn
        n_estimators : int, optional (default: 10)
            Number of trees to fit
        node_size : int, optional (default: 2)
            Maximum number of literals in each node.
        max_tree_depth : int, optional (default: 3)
            Maximum number of nodes from root to leaf (height) in the tree.
        neg_pos_ratio : int or float, optional (default: 2)
            Ratio of negative to positive examples used during learning.
        utility_alpha : float, optional (default: 1)
            Alpha hyperparameter for alpha-fairness utility function.
            
        Attributes
        ----------
        estimators_ : array, shape (n_estimators)
            Return the boosted regression trees
        feature_importances_ : array, shape (n_features)
            Return the feature importances (based on how often each feature appears)
        """

        super().__init__(
            background = background,
            target = target,
            n_estimators = n_estimators,
            node_size = node_size,
            max_tree_depth = max_tree_depth,
            neg_pos_ratio = neg_pos_ratio,
            solver = "BoostSRLTransferLearner",
            path = path
        )
        self.utility_alpha = utility_alpha

    def _check_params(self):
        if not (self.utility_alpha >= 0):
            raise ValueError("Alpha for alpha-fairness should be greater than or equal to zero.")
        super()._check_params()

    def fit(self, database, ignoreSTDOUT = True):
        self._check_params()

        if not isinstance(database, TransferLearningDatabase):
            raise ValueError("Database should be an instance of TransferLearningDatabase class.")

        # Write the background to file.
        self.background.write(
            filename="train", location=self.file_system.files.TRAIN_DIR
        )

        # Write the data to files.
        database.write(
            filename="train", location=self.file_system.files.TRAIN_DIR
        )

        _jar = str(self.file_system.files.BOOSTSRL_TRANFER_LEARNER_BACKEND)

        _CALL = (
            "java -jar "
            + _jar
            + " -l -train "
            + str(self.file_system.files.TRAIN_DIR)
            + " -target "
            + self.target
            + " -trees "
            + str(self.n_estimators)
            + " -negPosRatio "
            + str(self.neg_pos_ratio)
            + " -utilityAlpha "
            + str(self.utility_alpha)
            + " 2>&1 | tee "
            + str(self.file_system.files.TRAIN_LOG)
        )

        # Call the constructed command.
        self._call_shell_command(_CALL, ignoreSTDOUT = ignoreSTDOUT)

        # Read the trees from files.
        _estimators = []
        for _tree_number in range(self.n_estimators):
            with open(
                self.file_system.files.TREES_DIR.joinpath(
                    "{0}Tree{1}.tree".format(self.target, _tree_number)
                )
            ) as _fh:
                _estimators.append(_fh.read())

        self._get_dotfiles()
        self.estimators_ = _estimators

        # TODO: Get training time. It can be extracted from the training output file. Where can we store this metric? Maybe we can store in `self._prediction_metrics`. In that case, is it better to change the name of this attribute to something like `_metrics`?

        return self

    def _run_inference(self, database, ignoreSTDOUT = True) -> None:
        """Run inference mode on the BoostSRL Jar files.

        This is a helper method for ``self.predict`` and ``self.predict_proba``
        """

        self._check_initialized()

        # Write the background to file.
        self.background.write(
            filename="test", location=self.file_system.files.TEST_DIR
        )

        # Write the data to files.
        database.write(filename="test", location=self.file_system.files.TEST_DIR)
       
        _jar = str(self.file_system.files.BOOSTSRL_TRANFER_LEARNER_BACKEND)

        _CALL = (
            "java -jar "
            + _jar
            + " -i -test "
            + str(self.file_system.files.TEST_DIR)
            + " -model "
            + str(self.file_system.files.MODELS_DIR)
            + " -target "
            + self.target
            + " -trees "
            + str(self.n_estimators)
            + " -aucJarPath "
            + str(self.file_system.files.AUC_JAR)
            + " 2>&1 | tee "
            + str(self.file_system.files.TEST_LOG)
        )

        self._call_shell_command(_CALL, ignoreSTDOUT = ignoreSTDOUT)

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

        _results_db = self.file_system.files.TEST_DIR.joinpath(
            "results_" + self.target + ".db"
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