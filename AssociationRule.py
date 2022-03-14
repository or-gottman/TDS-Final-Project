
class AssociationRule:
    def __init__(self, left_side, right_side):
        self.left_side = left_side
        self.right_side = right_side
        self.support = 0
        self.confidence = 0

    def set_support(self, support_value):
        self.support = support_value

    def set_confidence(self, confidence_value):
        self.confidence = confidence_value
