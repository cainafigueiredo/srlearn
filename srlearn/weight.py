from abc import ABC, abstractmethod
from typing import Literal

class WeightStrategyBase(ABC):
    @abstractmethod
    def getWeights(self, examples, **kwargs):
        pass

class BalancedInstanceGroupUniformWeight(WeightStrategyBase):
    def __init__(
        self, 
        balanceStrength: float = 1
    ):
        assert balanceStrength >= 0 and balanceStrength <= 1, "`balanceStrength`should be a float in [0,1]."
        self.balanceStrength = balanceStrength

    def getWeights(self, examples, **kwargs):
        totalSourcePosExamples = len([
            example for example in examples if (example["domain"] == "sourceDomain" and example["label"] == "pos")
        ])
        totalSourceNegExamples = len([
            example for example in examples if (example["domain"] == "sourceDomain" and example["label"] == "neg")
        ])
        totalTargetPosExamples = len([
            example for example in examples if (example["domain"] == "targetDomain" and example["label"] == "pos")
        ])
        totalTargetNegExamples = len([
            example for example in examples if (example["domain"] == "targetDomain" and example["label"] == "neg")
        ])

        maxTotalExamples = max(
            totalSourcePosExamples, totalSourceNegExamples, totalTargetPosExamples, totalTargetNegExamples
        )

        totalSourcePosWeights = self.balanceStrength * maxTotalExamples + (1-self.balanceStrength) * totalSourcePosExamples

        totalSourceNegWeights = self.balanceStrength * maxTotalExamples + (1-self.balanceStrength) * totalSourceNegExamples

        totalTargetPosWeights = self.balanceStrength * maxTotalExamples + (1-self.balanceStrength) * totalTargetPosExamples

        totalTargetNegWeights = self.balanceStrength * maxTotalExamples + (1-self.balanceStrength) * totalTargetNegExamples

        scaleFactor = 1

        weights = IntraInstanceGroupUniformWeight(
            totalSourcePosWeights = totalSourcePosWeights,
            totalSourceNegWeights = totalSourceNegWeights,
            totalTargetPosWeights = totalTargetPosWeights,
            totalTargetNegWeights = totalTargetNegWeights,
            scaleFactor = scaleFactor
        ).getWeights(examples, **kwargs)
        
        return weights

class IntraInstanceGroupUniformWeight(WeightStrategyBase):
    """This weighting strategy assigns weights to each instance based on its group. There are 4 disjoint groups of instances: source pos, source neg, target pos, target neg. Each group has a total amount of weights to divide between its instances. Weights are divided evenly between instances of the same group. All weights can be scaled by a common factor."""

    def __init__(
        self, 
        totalSourcePosWeights: float = 1, 
        totalSourceNegWeights: float = 1, 
        totalTargetPosWeights: float = 1, 
        totalTargetNegWeights: float = 1,
        scaleFactor: float = 1
    ):
        self.totalSourcePosWeights = totalSourcePosWeights
        self.totalSourceNegWeights = totalSourceNegWeights
        self.totalTargetPosWeights = totalTargetPosWeights
        self.totalTargetNegWeights = totalTargetNegWeights
        self.scaleFactor = scaleFactor

    def getWeights(self, examples, **kwargs):
        totalSourcePosExamples = len([
            example for example in examples if (example["domain"] == "sourceDomain" and example["label"] == "pos")
        ])
        totalSourceNegExamples = len([
            example for example in examples if (example["domain"] == "sourceDomain" and example["label"] == "neg")
        ])
        totalTargetPosExamples = len([
            example for example in examples if (example["domain"] == "targetDomain" and example["label"] == "pos")
        ])
        totalTargetNegExamples = len([
            example for example in examples if (example["domain"] == "targetDomain" and example["label"] == "neg")
        ])
        sourcePosWeight = (self.totalSourcePosWeights / totalSourcePosExamples) * self.scaleFactor
        sourceNegWeight = (self.totalSourceNegWeights / totalSourceNegExamples) * self.scaleFactor
        targetPosWeight = (self.totalTargetPosWeights / totalTargetPosExamples) * self.scaleFactor
        targetNegWeight = (self.totalTargetNegWeights / totalTargetNegExamples) * self.scaleFactor
        weights = []
        for example in examples:
            if example["domain"] == "sourceDomain":
                if example["label"] == "pos":
                    weight = sourcePosWeight
                else:
                    weight = sourceNegWeight
            else:
                if example["label"] == "pos":
                    weight = targetPosWeight
                else:
                    weight = targetNegWeight
            weights.append(weight)
        return weights

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
        assert strategy in set([
            "balancedInstanceGroupUniform", "intraDomainUniform", "uniform", "scalar"
        ]), f"`{strategy}` is not a valid weight strategy."

        if strategy == "balancedInstanceGroupUniform":
            balanceStrength = kwargs.get("balanceStrength", 1)
            return self.getBalancedInstanceGroupUniformWeight(balanceStrength)

        elif strategy == "intraInstanceGroupUniform":
            totalSourcePosWeights = kwargs.get("totalSourcePosWeights", 1)
            totalSourceNegWeights = kwargs.get("totalSourceNegWeights", 1)
            totalTargetPosWeights = kwargs.get("totalTargetPosWeights", 1)
            totalTargetNegWeights = kwargs.get("totalTargetNegWeights", 1)
            scaleFactor = kwargs.get("scaleFactor", 1)
            return self.getIntraInstanceGroupUniformWeight(
                totalSourcePosWeights,
                totalSourceNegWeights,
                totalTargetPosWeights,
                totalTargetNegWeights,
                scaleFactor
            )

        elif strategy == "intraDomainUniform":
            targetTotalWeightFraction = kwargs.get("targetTotalWeightFraction", None)
            return self.getIntraDomainUniformWeight(targetTotalWeightFraction = targetTotalWeightFraction)
        
        elif strategy == "uniform":
            return self.getUniformWeight()
        
        elif strategy == "scalar":
            weight = kwargs.get("weight", None)
            return self.getScalarWeight(weight = weight)
    
    def getBalancedInstanceGroupUniformWeight(self, balanceStrength = 1):
        return BalancedInstanceGroupUniformWeight(balanceStrength)

    def getIntraInstanceGroupUniformWeight(
        self,
        totalSourcePosWeights = 1,
        totalSourceNegWeights = 1,
        totalTargetPosWeights = 1,
        totalTargetNegWeights = 1,
        scaleFactor = 1
    ):
        return IntraInstanceGroupUniformWeight(
            totalSourcePosWeights,
            totalSourceNegWeights,
            totalTargetPosWeights,
            totalTargetNegWeights,
            scaleFactor
        )

    def getIntraDomainUniformWeight(self, targetTotalWeightFraction = None):
        assert targetTotalWeightFraction, f"`targetTotalWeightFraction` is not set."
        return IntraDomainUniformWeight(targetTotalWeightFraction = targetTotalWeightFraction)
    
    def getUniformWeight(self):
        return UniformWeight()
    
    def getScalarWeight(self, weight = None):
        assert weight, f"`weight` is not set."
        return ScalarWeight(weight = weight)