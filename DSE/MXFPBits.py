class MXFPBits:
  def __init__(self, exp_bits, mant_bits):
    self.exp_bits = exp_bits
    self.mant_bits = mant_bits
    
  def __repr__(self):
    return f"E{self.exp_bits}M{self.mant_bits}"