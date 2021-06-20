import pandas as pd
import pytest

from autoarm import Dataset, FrequentItemsets

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
    assert type(frequent_itemsets.to_frame()) == pd.core.frame.DataFrame

    min_support = 0.01
    frequent_itemsets = FrequentItemsets(dataset, min_support)
    assert type(frequent_itemsets.to_frame()) == pd.core.frame.DataFrame

    min_support = 0.5
    frequent_itemsets = FrequentItemsets(dataset, min_support)
    assert type(frequent_itemsets.to_frame()) == pd.core.frame.DataFrame

    min_support = 1
    frequent_itemsets = FrequentItemsets(dataset, min_support)
    assert type(frequent_itemsets.to_frame()) == pd.core.frame.DataFrame

    with pytest.raises(ValueError):
        min_support = -1
        frequent_itemsets = FrequentItemsets(dataset, min_support)

    with pytest.raises(ValueError):
        min_support = 0
        frequent_itemsets = FrequentItemsets(dataset, min_support)

    with pytest.raises(ValueError):
        min_support = 2
        frequent_itemsets = FrequentItemsets(dataset, min_support)
