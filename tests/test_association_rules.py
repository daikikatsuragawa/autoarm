import pandas as pd
import pytest

from autoarm import AssociationRules, Dataset, FrequentItemsets

sample_dataset = {
    "transaction_id": 
    [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8],
    "item_id": [
        "A", "B", "C", "D", "A", "B", "C", "A", "B", "A", "E", "F", "B", "C", "D",
        "F","B","C", "F", "B", "A", "E"
    ],
}


def test():
    sample_df = pd.DataFrame.from_dict(sample_dataset)
    transaction_column = "transaction_id"
    item_column = "item_id"
    dataset = Dataset(sample_df, transaction_column, item_column)
    frequent_itemsets = FrequentItemsets(dataset)

    metric = "confidence"
    min_confidence = 0
    association_rules = AssociationRules(frequent_itemsets,
                                         metric=metric,
                                         min_threshold=min_confidence)
    assert type(association_rules.to_frame()) == pd.core.frame.DataFrame

    metric = "confidence"
    min_confidence = 1
    association_rules = AssociationRules(frequent_itemsets,
                                         metric=metric,
                                         min_threshold=min_confidence)
    assert type(association_rules.to_frame()) == pd.core.frame.DataFrame

    metric = "lift"
    min_lift = 0
    association_rules = AssociationRules(frequent_itemsets,
                                         metric=metric,
                                         min_threshold=min_lift)
    assert type(association_rules.to_frame()) == pd.core.frame.DataFrame

    metric = "lift"
    min_lift = 1
    association_rules = AssociationRules(frequent_itemsets,
                                         metric=metric,
                                         min_threshold=min_lift)
    assert type(association_rules.to_frame()) == pd.core.frame.DataFrame

    with pytest.raises(ValueError):
        metric = "other"
        association_rules = AssociationRules(frequent_itemsets,
                                             metric=metric,
                                             min_threshold=min_confidence)

    with pytest.raises(ValueError):
        min_confidence = -1
        metric = "confidence"
        association_rules = AssociationRules(frequent_itemsets,
                                             metric=metric,
                                             min_threshold=min_confidence)

    with pytest.raises(ValueError):
        min_confidence = 2
        metric = "confidence"
        association_rules = AssociationRules(frequent_itemsets,
                                             metric=metric,
                                             min_threshold=min_confidence)

    with pytest.raises(ValueError):
        min_lift = -1
        metric = "lift"
        association_rules = AssociationRules(frequent_itemsets,
                                             metric=metric,
                                             min_threshold=min_lift)
