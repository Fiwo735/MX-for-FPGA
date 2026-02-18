from enum import Enum

class AccumMethod(Enum):
  Kulisch = "KULISCH"
  Kahan = "KAHAN"
  Neumaier = "NEUMAIER"
  Klein = "KLEIN"
  TwoSum = "TWOSUM"
  FastTwoSum = "FASTTWOSUM"
  Naive = "NAIVE"
  Quant = "QUANT"