from abc import ABC, abstractmethod
from typing import Literal

class WeightStrategyBase(ABC):
    @abstractmethod
    def getWeights(self, examples, **kwargs):
        pass

class IntraDomainUniformWeight(WeightStrategyBase):
    def __init__(self, targetTotalWeightFraction: float):
        assert 0 <= targetTotalWeightFraction <= 1
        self.targetTotalWeightFraction = targetTotalWeightFraction

    def getWeights(self, examples, **kwargs):
        totalSourceExamples = len([example for example in examples if example["domain"] == "sourceDomain"])
        totalTargetExamples = len([example for example in examples if example["domain"] == "targetDomain"])
        targetTargetWeight = self.targetTotalWeightFraction/totalTargetExamples
        targetSourceWeight = (1-self.targetTotalWeightFraction)/totalSourceExamples
        weights = [targetSourceWeight if example["domain"] == "sourceDomain" else targetTargetWeight for example in examples]
        return weights

class UniformWeight(WeightStrategyBase):
    def __init__(self):
        pass

    def getWeights(self, examples, **kwargs):
        totalExamples = len(examples)
        weight = 1/totalExamples
        weights = [weight]*totalExamples
        return weights
    
class ScalarWeight(WeightStrategyBase):
    def __init__(self, weight = 1) -> None:
        self.weight = weight

    def getWeights(self, examples, **kwargs):
        totalExamples = len(examples)
        weights = [self.weight]*totalExamples
        return weights

class WeightFactory:
    def __init__(self):
        pass
        
    def getWeightStrategy(self, strategy: Literal["intraDomainUniform", "uniform", "scalar"], **kwargs) -> None:
        assert strategy in set(["intraDomainUniform", "uniform", "scalar"]), f"`{strategy}` is not a valid weigth strategy."

        if strategy == "intraDomainUniform":
            targetTotalWeightFraction = kwargs.get("targetTotalWeightFraction", None)
            return self.getIntraDomainUniformWeight(targetTotalWeightFraction = targetTotalWeightFraction)
        
        elif strategy == "uniform":
            return self.getUniformWeight()
        
        elif strategy == "scalar":
            weight = kwargs.get("weight", None)
            return self.getScalarWeight(weight = weight)
            
    def getIntraDomainUniformWeight(self, targetTotalWeightFraction = None):
        assert targetTotalWeightFraction, f"`targetTotalWeightFraction` is not set."
        return IntraDomainUniformWeight(targetTotalWeightFraction = targetTotalWeightFraction)
    
    def getUniformWeight(self):
        return UniformWeight()
    
    def getScalarWeight(self, weight = None):
        assert weight, f"`weight` is not set."
        return ScalarWeight(weight = weight)