# Copyright 2017, 2018, 2019 Alexander L. Hayes

"""
database.py

A BoostSRL database consists of positive examples, negative examples, and facts;
all of which need to be stored as .txt files on a file system.

Use Cases
---------

- Creating an instance of the database through code (write to location)
- Files already stored on the filesystem (copy to location)
- Examples stored in a RDBMS?

Examples
--------

Create a new instance of a database, add examples, and write them to the filesystem.

>>> from srlearn.database import Database
>>> db = Database()
>>> db.add_pos("student(alexander).")
>>> db.add_neg("student(sriraam).")
>>> db.add_fact("advises(alexander, sriraam).")

Create an instance of the database from an existing set of files.

>>> from srlearn.database import Database
>>> db = Database()
"""

from shutil import copyfile
import numpy as np
import pathlib
import copy
import re

from typing import Type, Dict, List, Tuple
from .utils import utils

class Database:
    """Database of examples and facts."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        """Initialize a Database object

        A database (in this respect) contains positive examples, negative examples,
        facts, and is augmented with background knowledge.

        The implementation is done with four attributes: ``pos``, ``neg``,
        ``facts``, and ``modes``. Each attribute is a list that may be set by
        mutating, or loaded from files with :func:`Database.from_files`.

        Examples
        --------

        This initializes a Database object, then sets the ``pos`` attribute.

        >>> from srlearn import Database
        >>> db = Database()
        >>> db.pos = ["student(alexander)."]
        """
        self.pos = []
        self.neg = []
        self.facts = []
        self.modes = []

    def write(self, filename: str = "train", location: pathlib.Path = pathlib.Path("train")) -> None:
        """Write the database to disk

        Parameters
        ----------
        filename : str
            Name of the file to write to: 'train' or 'test'
        location : :class:`pathlib.Path`
            Path where data should be written to.

        Notes
        -----

        This function has polymorphic behavior. When attributes (``self.pos``,
        ``self.neg``, ``self.facts``) are lists of strings, the lists are
        written to files. When the attributes are (path-like) strings or
        pathlib Paths (:class:`pathlib.Path`), the files are copied.
        """

        def _write(_filename, _location, _object, _type):
            if isinstance(_object, list):
                with open(
                    _location.joinpath("{0}_{1}.txt".format(_filename, _type)), "w"
                ) as _fh:
                    for example in _object:
                        _fh.write(example + "\n")
            else:
                copyfile(
                    str(_object),
                    str(_location.joinpath("{0}_{1}.txt".format(_filename, _type))),
                )

        location.mkdir(exist_ok = True)

        _write(filename, location, self.pos, "pos")
        _write(filename, location, self.neg, "neg")
        _write(filename, location, self.facts, "facts")

    def __repr__(self) -> str:
        return (
            "Positive Examples:\n"
            + str(self.pos)
            + "\nNegative Examples:\n"
            + str(self.neg)
            + "\nFacts:\n"
            + str(self.facts)
        )

    @staticmethod
    def fromFiles(
        pos: str = "pos.pl", 
        neg: str = "neg.pl", 
        facts: str = "facts.pl", 
        modes: str = "modes.pl", 
        useRecursion: bool = True
    ):
        """Load files into a Database

        Return an instance of a Database with pos, neg, facts, and modes set to the
        contents of files. By default this performs a "lazy load," where the
        files are not loaded into Python lists, but copied at learning time.

        Parameters
        ----------
        pos : str or pathlib.Path
            Location of positive examples
        neg : str or pathlib.Path
            Location of negative examples
        facts : str or pathlib.Path
            Location of facts
        modes : str or pathlib.Path
            Location of modes
        lazy_load : bool (default: True)
            Skip loading the files into a list

        Returns
        -------
        db : srlearn.Database
            Instance of a Database object
        """

        _db = Database()

        with open(pos, "r") as _fh:
            _db.pos = _fh.read().splitlines()
        with open(neg, "r") as _fh:
            _db.neg = _fh.read().splitlines()
        with open(facts, "r") as _fh:
            _db.facts = _fh.read().splitlines()
        with open(modes, "r") as _fh:
            _db.modes = _fh.read().splitlines()

        if useRecursion:
            _db = _db.generateRecursion()
        else:
            _db = _db.removeRecursion()

        return _db

    def applyMapping(self, relationMapping: Dict, termTypeMapping: Dict, termPrefix: str = None) -> Type["Database"]:
        # Mapping facts
        facts = []
        for fact in self.facts:
            relation, terms = re.findall(r"(.*)\((.*)\)\.", fact)[0]
            if relationMapping.get(relation) is not None:
                if termPrefix:
                    terms = [f"{termPrefix}{term.strip()}" for term in terms.split(",")]
                    terms = ",".join(terms)
                relation = relationMapping.get(relation)
                literal = f"{relation}({terms})."
                facts.append(literal)

        # Mapping positive examples
        pos = []
        for example in self.pos:
            relation, terms = re.findall(r"(.*)\((.*)\)\.", example)[0]
            if termPrefix:
                terms = [f"{termPrefix}{term.strip()}" for term in terms.split(",")]
                terms = ",".join(terms)
            relation = relationMapping.get(relation)
            literal = f"{relation}({terms})."
            pos.append(literal)

        # Mapping negative examples
        neg = []
        for example in self.neg:
            relation, terms = re.findall(r"(.*)\((.*)\)\.", example)[0]
            if termPrefix:
                terms = [f"{termPrefix}{term.strip()}" for term in terms.split(",")]
                terms = ",".join(terms)
            relation = relationMapping.get(relation)
            literal = f"{relation}({terms})."
            neg.append(literal)

        # Mapping modes
        modes = []
        for mode in self.modes:
            relation, terms = re.findall(r"(.*)\((.*)\)\.", mode)[0]
            if relationMapping.get(relation) is not None:
                terms = [term.strip() for term in terms.split(",")]
                for i, term in enumerate(terms):
                    termTypeBeforeMapping = re.sub(r"[\+\-\#\s]", "", term)
                    termTypeAfterMapping = termTypeMapping.get(termTypeBeforeMapping)
                    terms[i] = term.replace(termTypeBeforeMapping, termTypeAfterMapping)
                terms = ",".join(terms)
                relation = relationMapping.get(relation)
                literal = f"{relation}({terms})."
                modes.append(literal)

        mappedDatabase = Database()
        mappedDatabase.facts = facts
        mappedDatabase.pos = pos
        mappedDatabase.neg = neg
        mappedDatabase.modes = modes

        return mappedDatabase
    
    def getTargetRelation(self):
        if len(self.pos) > 0:
            relation = re.findall(r"(.*)\(.*\)\.", self.pos[0])[0]
        else:
            relation = None
        return relation

    def extractSchemaPreds(self) -> Dict:
        schema = {}
        for mode in self.modes:
            relation, termsTypes = re.findall(r"(.*)\((.*)\)\.", mode)[0]
            termsTypes = [re.sub(r"[\+\-\#\s]", "", termType.strip()) for termType in termsTypes.split(",")]
            schema[relation] = termsTypes
        return schema

    @staticmethod
    def _findAllValidMappingsRecursive(
        sourceSchemaPreds: Dict,
        targetSchemaPreds: Dict,
        relationMapping: Dict = {}, 
        termTypeMapping: Dict = {}
    ) -> List[Tuple[Dict, Dict]]:
        if len(sourceSchemaPreds) == 0:
            return [(relationMapping, termTypeMapping)]
        
        sourcePredicateToMapRelation, sourcePredicateToMapTermTypes = list(sourceSchemaPreds.items())[0]
        newSourceSchemaPreds = {**sourceSchemaPreds}
        del newSourceSchemaPreds[sourcePredicateToMapRelation]

        mappings = []

        targetSchemaPredsWithEmpty = {**targetSchemaPreds, None: None}.items()
        for targetDomainCandidatePredicateRelation, targetDomainCandidatePredicateTermTypes in targetSchemaPredsWithEmpty:
            if utils.isCompatible(
                sourcePredicateToMapRelation, 
                targetDomainCandidatePredicateRelation, 
                sourcePredicateToMapTermTypes,
                targetDomainCandidatePredicateTermTypes,
                termTypeMapping,
            ):
                newRelationMapping = copy.deepcopy(relationMapping)
                newTermTypeMapping = copy.deepcopy(termTypeMapping)
                
                # If it is not mapping a source predicate to an empty predicate
                if targetDomainCandidatePredicateRelation is not None:
                    newRelationMapping[sourcePredicateToMapRelation] = targetDomainCandidatePredicateRelation
                    for sourceTermType, targetTermType in zip(
                        sourcePredicateToMapTermTypes, targetDomainCandidatePredicateTermTypes
                    ): 
                        newTermTypeMapping[sourceTermType] = targetTermType

                # If it is mapping a source predicate to an empty predicate
                else: 
                    newRelationMapping[sourcePredicateToMapRelation] = None

                newTargetSchemaPreds = {**targetSchemaPreds}
                if targetDomainCandidatePredicateRelation is not None:
                    del newTargetSchemaPreds[targetDomainCandidatePredicateRelation]

                mappings += Database._findAllValidMappingsRecursive(
                    newSourceSchemaPreds, 
                    newTargetSchemaPreds, 
                    newRelationMapping, 
                    newTermTypeMapping
                )

        return mappings

    def removeAllPredicatesOfRelation(self, relationName: str) -> Type["Database"]:
        database = copy.deepcopy(self)
        database.facts = utils.removeAllPredicatesOfRelation(relationName, database.facts)
        database.pos = utils.removeAllPredicatesOfRelation(relationName, database.pos)
        database.neg = utils.removeAllPredicatesOfRelation(relationName, database.neg)
        database.modes = utils.removeAllPredicatesOfRelation(relationName, database.modes)
        return database

    def findAllValidMappings(
        self,
        targetDatabase: Type["Database"]
    ) -> List[Tuple[Dict, Dict]]:
        # TODO: what to handle mapping for recursion predicates? Suggestion: Map only the target predicate. Then, recursion relations of both domains are automatically mapped. I will probably need to modify the method applyMapping. I can remove recursion predicates from the list of relations to be mapped and then include a mapping between the recursions of target predicates from both domains.
        sourceDatabase = self
        sourceSchemaPreds = sourceDatabase.extractSchemaPreds()
        targetSchemaPreds = targetDatabase.extractSchemaPreds()
        targetDomainTargetRelation = targetDatabase.getTargetRelation()
        targetDomainTargetPredicateTermTypes = targetSchemaPreds[targetDomainTargetRelation]
        newTargetSchemaPreds = {**targetSchemaPreds}
        del newTargetSchemaPreds[targetDomainTargetRelation]
        allMappings = []
        for candidateSourceRelation, candidateSourcePredicateTermTypes in sourceSchemaPreds.items():
            if utils.isCompatible(
                candidateSourceRelation, 
                targetDomainTargetRelation, 
                candidateSourcePredicateTermTypes,
                targetDomainTargetPredicateTermTypes,
                {}
            ):
                relationMapping = {candidateSourceRelation: targetDomainTargetRelation}
                termTypeMapping = dict(zip(candidateSourcePredicateTermTypes, targetDomainTargetPredicateTermTypes))
                newSourceSchemaPreds = {**sourceSchemaPreds}
                del newSourceSchemaPreds[candidateSourceRelation]
                mappings = Database._findAllValidMappingsRecursive(
                    newSourceSchemaPreds, 
                    newTargetSchemaPreds,
                    relationMapping = relationMapping,
                    termTypeMapping = termTypeMapping
                )
                allMappings += mappings
        return allMappings
    
    def resetTargetPredicate(self) -> Type["Database"]:
        """Move positive examples of target predicate to facts and empty positive and negative examples."""
        database = copy.deepcopy(self)
        targetRelation = self.getTargetRelation()
        database.facts = utils.removeAllPredicatesOfRelation(f"recursion_{targetRelation}", database.facts)
        database.facts += database.pos
        database.pos = []
        database.neg = []
        database.modes = utils.removeAllPredicatesOfRelation(f"recursion_{targetRelation}", database.modes)
        return database

    def getTotalPositiveExamples(self):
        return len(self.pos)
    
    def getTotalNegativeExamples(self):
        return len(self.neg)
    
    def getTotalFacts(self):
        return len(self.facts)

    def removeRecursion(self) -> Type["Database"]:
        targetRelation = self.getTargetRelation()
        if not targetRelation:
            raise Exception("Target relation is undefined.")
        recursionRelation = f"recursion_{targetRelation}"
        database = copy.deepcopy(self)
        database.modes = utils.removeAllPredicatesOfRelation(recursionRelation, database.modes)
        database.facts = utils.removeAllPredicatesOfRelation(recursionRelation, database.facts)
        return database

    def generateRecursion(self) -> Type["Database"]:
        database = self.removeRecursion()
        targetRelation = database.getTargetRelation()
        recursionRelation = f"recursion_{targetRelation}"
        mapping = {targetRelation: recursionRelation}
        schema = database.extractSchemaPreds()
        for i in range(len(schema[targetRelation])):
            arguments = [f"-{argumentType}" if j == i else f"+{argumentType}" for j, argumentType in enumerate(schema[targetRelation])]
            mode = f"{recursionRelation}({','.join(arguments)})."
            database.modes.append(mode)
        database.facts += utils.renameRelationsInPredicates(database.pos, mapping = mapping)
        return database

    def setTargetPredicate(self, relationName, useRecursion = True, negPosRatio = 1, seed = None, maxFailedNegSamplingRetries = 50) -> Type["Database"]:
        """Move positive examples of target predicate to facts and move facts of 'relationName' to positive examples list. Then, it samples negative examples."""
        assert type(negPosRatio) is int
        database = self.resetTargetPredicate()
        database.pos = utils.findAllPredicatesOfRelation(relationName, database.facts)
        database.facts = utils.removeAllPredicatesOfRelation(relationName, database.facts)
        if useRecursion:
            database = database.generateRecursion()
        database.neg = database.generateNegativePredicates(negPosRatio, seed, maxFailedNegSamplingRetries)
        return database

    def getClosedWorldConstants(self):
        """It extracts all possible instantiations for each of object type in the dataset."""

        domainSchemaPreds = self.extractSchemaPreds()
        allPredicates = self.facts + self.pos

        objectInstances = {}

        for predicate in allPredicates:
            relation = utils.getPredicateRelationName(predicate)
            if relation in domainSchemaPreds:
                objects = utils.getPredicateTerms(predicate)
                for i, obj in enumerate(objects):
                    objType = domainSchemaPreds[relation][i]
                    if objType not in objectInstances:
                        objectInstances[objType] = set([])
                    objectInstances[objType].add(obj)
        
        objectInstances = {k:list(v) for k,v in objectInstances.items()}

        return objectInstances
    
    def generateNegativePredicates(self, negPosRatio = 1, seed = None, maxFailedSamplingRetries = 50):
        '''Receives positive examples and generates all negative examples'''
        targetPredicateRelation = self.getTargetRelation()
        targetPredicateTermsTypes = self.extractSchemaPreds()[targetPredicateRelation]
        positiveExamplesTerms = set([tuple(utils.getPredicateTerms(predicate)) for predicate in self.pos])
        objectsInstances = {k:v for k, v in self.getClosedWorldConstants().items() if k in targetPredicateTermsTypes}
        
        totalNegativesToSample = negPosRatio * self.getTotalPositiveExamples()

        np.random.seed(seed)

        negativeExamples = set([])
        totalNegativeExamples = 0
        successiveFailedSampling = 0
        shouldExitLoop = False
        while totalNegativeExamples < totalNegativesToSample:
            sample = None
            while sample in positiveExamplesTerms or sample in negativeExamples or (sample is None):
                if successiveFailedSampling >= maxFailedSamplingRetries:
                    shouldExitLoop = True
                    break
                sample = tuple([np.random.choice(objectsInstances[objType], 1)[0] for objType in targetPredicateTermsTypes])
                successiveFailedSampling += 1
            if shouldExitLoop:
                break
            negativeExamples.add(sample)
            totalNegativeExamples += 1
            successiveFailedSampling = 0

        negativeExamples = [f"{targetPredicateRelation}({','.join(negativeExample)})." for negativeExample in negativeExamples]

        np.random.seed(None)

        return negativeExamples

    def isCompatibleWithDatabase(self, database: Type["Database"]):
        selfSchemaPreds = set(
            [tuple([relation, tuple(termsTypes)]) for relation, termsTypes in self.extractSchemaPreds().items()]
        )
        otherDatabaseSchemaPreds = set(
            [tuple([relation, tuple(termsTypes)]) for relation, termsTypes in database.extractSchemaPreds().items()]
        )
        unmappedSourcePreds = selfSchemaPreds - otherDatabaseSchemaPreds
        
        if unmappedSourcePreds != set():
            errorMessage = "The following predicates from source database schema do not exist in the target database schema: {}.".format([f"{relation}({','.join(termsTypes)})." for relation, termsTypes in unmappedSourcePreds])
            raise Exception(errorMessage)
        
        selfTargetPredicate = self.getTargetRelation()
        otherDatabaseSchemaPredsTargetPredicate = database.getTargetRelation()
        if selfTargetPredicate != otherDatabaseSchemaPredsTargetPredicate:
            raise Exception("The target predicate should be the same for both domain.")
        
        return True

    def merge(
        self,
        targetDatabase: Type["Database"]
    ):        
        sourceDatabase = self
        if self.isCompatibleWithDatabase(targetDatabase):
            mergedDatabase = Database()
            mergedDatabase.modes = targetDatabase.modes
            mergedDatabase.facts = targetDatabase.facts + sourceDatabase.facts
            mergedDatabase.pos = targetDatabase.pos + sourceDatabase.pos
            mergedDatabase.neg = targetDatabase.neg + sourceDatabase.neg
            return mergedDatabase
    
    @staticmethod
    def prepareTransferLearningDatabase(
            sourceDatabase: Type["Database"], targetDatabase: Type["Database"]
    ) -> Type["TransferLearningDatabase"]:
        if sourceDatabase.isCompatibleWithDatabase(targetDatabase):
            database = TransferLearningDatabase(sourceDatabase, targetDatabase)
            return database
        
class TransferLearningDatabase(Database):
    def __init__(self, sourceDatabase: Type["Database"], targetDatabase: Type["Database"]):
        super(TransferLearningDatabase, self).__init__()
        self.facts = sourceDatabase.facts + targetDatabase.facts
        self.pos = sourceDatabase.pos + targetDatabase.pos
        self.neg = sourceDatabase.neg + targetDatabase.neg
        self.modes = targetDatabase.modes
        self.posExampleDomain = ["sourceDomain" for i in sourceDatabase.pos] + ["targetDomain" for i in targetDatabase.pos]
        self.negExampleDomain = ["sourceDomain" for i in sourceDatabase.neg] + ["targetDomain" for i in targetDatabase.neg]

    def write(self, filename: str = "train", location: pathlib.Path = pathlib.Path("train")) -> None:
        def _write(_filename, _location, _type):
            with open(
                _location.joinpath("{0}_{1}.txt".format(_filename, _type)), "w"
            ) as _fh:
                
                if _type == "pos":
                    for exampleDomain, example in zip(self.posExampleDomain, self.pos):
                        _fh.write(f"{exampleDomain}: {example}\n")

                elif _type == "neg":
                    for exampleDomain, example in zip(self.negExampleDomain, self.neg):
                        _fh.write(f"{exampleDomain}: {example}\n")

                elif _type == "facts":
                    for example in self.facts:
                        _fh.write(f"{example}\n")

        location.mkdir(exist_ok = True)

        _write(filename, location, "pos")
        _write(filename, location, "neg")
        _write(filename, location, "facts")