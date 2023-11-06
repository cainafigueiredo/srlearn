import re
import random
import copy

def isCompatible(sourceRelation, targetRelation, sourceTermsTypes, targetTermsTypes, termTypeMapping):
    '''Determines if arguments mapping is compatible or not'''
    if sourceRelation is None or sourceRelation in termTypeMapping:
        return False
    
    if targetRelation is None:
        return True
    
    tmpTermTypeMapping = {**termTypeMapping}

    if len(sourceTermsTypes) != len(targetTermsTypes):
        return False
    
    for termType1, termType2 in zip(sourceTermsTypes, targetTermsTypes):
        if termType1 in tmpTermTypeMapping and tmpTermTypeMapping[termType1] != termType2:
            return False
        else:
            tmpTermTypeMapping[termType1] = termType2

    return True

def getPredicateTerms(predicate):
    termTypes = [term.strip() for term in re.findall(r".*\((.*)\)\.", predicate)[0].split(",")]
    return termTypes

def getPredicateRelationName(predicate):
    relation = re.findall(r"(.*)\(.*\)\.", predicate)[0].strip()
    return relation

def findAllPredicatesOfRelation(relationName, predicatesList):
    predicates = [predicate for predicate in predicatesList if getPredicateRelationName(predicate) == relationName]
    return predicates

def removeAllPredicatesOfRelation(relationName, predicatesList):
    predicates = [predicate for predicate in predicatesList if getPredicateRelationName(predicate) != relationName]
    return predicates

def samplePredicates(predicates, sampleSize, seed=None):
    '''Receives negative examples and balance them according to the number of positive examples'''
    predicates = copy.deepcopy(predicates)
    random.seed(seed)
    random.shuffle(predicates)
    predicates = predicates[:sampleSize]
    random.seed(None)
    return predicates

def renameRelationsInPredicates(predicates: list, mapping: dict):
    renamedPredicates = []
    for predicate in predicates:
        relation = getPredicateRelationName(predicate)
        if relation in mapping:
            newRelation = mapping[relation]
            terms = getPredicateTerms(predicate)
            predicate = f"{newRelation}({','.join(terms)})."
        renamedPredicates.append(predicate)
    return renamedPredicates