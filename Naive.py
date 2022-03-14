from AssociationRule import AssociationRule


class Naive:
    def __init__(self):
        self.generated_association_rules = list()

    def run(self):
        a = AssociationRule("a", "b")
