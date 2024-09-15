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

def cleanPreds(preds):
    '''Clean +, -, and # from modes'''
    ret = set()
    for line in preds:
        m = re.search('^(\w+)\(([\w, +\-\#\`]+)*\).$', line)
        if m:
            relation = m.group(1)
            relation = re.sub('[+\-\#\` ]', '', relation)
            entities = m.group(2)
            entities = re.sub('[+\-\#\` ]', '', entities)
            ret.add(relation + '(' + entities + ').')
    return list(ret)

def convertLiteralsToArity2(literals, model: dict = None):
    newLiterals = set()

    literalsAreModes = False

    if any([mode in literals[0] for mode in ["+", "-", "#", "`"]]):
        literalsAreModes = True

    if not literalsAreModes:
        assert model is not None, "Model is required to convert literals. You can extract it from a Database object by calling the method `extractSchemaPreds`."

    unaryLiterals = []
    nAryLiterals = []

    for literal in literals:
        predicate, terms = re.findall(r"(.*)\((.*)\)\.", literal)[0]
        terms = terms.split(",")
        arity = len(terms)
        if arity == 2:
            newLiterals.add(literal)
        elif arity == 1:
            unaryLiterals.append(literal)
        else:
            nAryLiterals.append(literal)

    unaryLiterals = cleanPreds(unaryLiterals)
    nAryLiterals = cleanPreds(nAryLiterals)

    for literal in unaryLiterals:
        predicate, term = re.findall(r"(.*)\((.*)\)\.", literal)[0]
        if literalsAreModes:
            for firstArgMode, secondArgMode in [("+", "+"), ("+", "-"), ("-", "+")]:
                newLiteral = f"{term}haslabel({firstArgMode}{term},{secondArgMode}{term}label)."
                newLiterals.add(newLiteral)
        else:
            termType = model[predicate][0]
            newLiteral = f"{termType}haslabel({term},{predicate})."
            newLiterals.add(newLiteral)

    nAryPredicateUniqueId = {}

    for literal in nAryLiterals:
        predicate, terms = re.findall(r"(.*)\((.*)\)\.", literal)[0]
        terms = terms.split(",")
        arity = len(terms)
        if literalsAreModes:
            for term in terms:
                for firstArgMode, secondArgMode in [("+", "+"), ("+", "-"), ("-", "+")]:
                    newLiteral = f"{predicate}{term}({firstArgMode}{predicate},{secondArgMode}{term})."
                    newLiterals.add(newLiteral)
        else:
            termTypes = model[predicate]
            if predicate not in nAryPredicateUniqueId:
                nAryPredicateUniqueId[predicate] = 1
            predicateUniqueId = nAryPredicateUniqueId[predicate]
            for term, termType in zip(terms, termTypes):
                newLiteral = f"{predicate}{termType}({predicate}{predicateUniqueId},{term})."
                newLiterals.add(newLiteral)
            nAryPredicateUniqueId[predicate] += 1
            
    return list(newLiterals)