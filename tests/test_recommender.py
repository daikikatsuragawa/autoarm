import pandas as pd
import pytest

from autoarm import AssociationRules, Dataset, FrequentItemsets, Recommender

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
    association_rules = AssociationRules(frequent_itemsets)

    items = ["A"]
    recommender = Recommender(association_rules)
    recommend_rules = recommender.recommend(items)
    assert type(recommend_rules) == pd.core.frame.DataFrame

    items = ["other"]
    recommender = Recommender(association_rules)
    recommend_rules = recommender.recommend(items)
    assert type(recommend_rules) == pd.core.frame.DataFrame

    recommend_rules = recommender.recommend(items,
                                            n=1,
                                            allow_from_items=True,
                                            metric="confidence")
    assert type(recommend_rules) == pd.core.frame.DataFrame

    with pytest.raises(ValueError):
        recommender = Recommender(association_rules, n=0)

    with pytest.raises(ValueError):
        metric = "other"
        recommender = Recommender(association_rules, metric=metric)

    with pytest.raises(ValueError):
        items = []
        recommender = Recommender(association_rules)
        recommend_rules = recommender.recommend(items)

    with pytest.raises(ValueError):
        items = ["A"]
        recommender = Recommender(association_rules)
        recommend_rules = recommender.recommend(items, n=0)

    with pytest.raises(ValueError):
        items = ["A"]
        metric = "other"
        recommender = Recommender(association_rules)
        recommend_rules = recommender.recommend(items, metric=metric)

    # min_support=0.01, metric="confidence", min_threshold=0
    sample_df = pd.DataFrame.from_dict(sample_dataset)
    dataset = Dataset(sample_df, transaction_column, item_column)
    frequent_itemsets = FrequentItemsets(dataset, min_support=0.01)
    assert len(frequent_itemsets.to_frame()) == 25

    association_rules = AssociationRules(frequent_itemsets, min_threshold=0)
    association_rules.to_frame()
    assert len(association_rules.to_frame()) == 90

    # min_support=0.01, metric="confidence", min_threshold=0
    sample_df = pd.DataFrame.from_dict(sample_dataset)
    dataset = Dataset(sample_df, transaction_column, item_column)
    frequent_itemsets = FrequentItemsets(dataset)
    assert len(frequent_itemsets.to_frame()) == 4

    association_rules = AssociationRules(frequent_itemsets)
    association_rules.to_frame()
    assert len(association_rules.to_frame()) == 1

def test_with_various_patterns_of_data():
    sample_dataset = {
        'transaction_id':
        [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7],
        'item_id': [
            "X", "Y", "Z", "X", "B", "Y", "A", "C", "A", "C", "X", "Y", "Z",
            "X", "Y", "B", "A", "X", "B"
        ],
    }
    df = pd.DataFrame.from_dict(sample_dataset)
    dataset = Dataset(df, "transaction_id", "item_id")
    frequent_itemsets = FrequentItemsets(dataset, min_support=0.01)
    association_rules = AssociationRules(frequent_itemsets,
                                         metric="confidence",
                                         min_threshold=0.1)

    # items size is one, metric is "confidence", allow_from_items is False
    recommender = Recommender(association_rules)

    items = ["X"]
    recommend_rules = recommender.recommend(items, n=6, metric="confidence")

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset("X"))
    assert (recommend_rules["consequents"][0] == frozenset(("B")))
    s = 0.428571
    c = 0.6
    l = 1.40
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset("X"))
    assert (recommend_rules["consequents"][1] == frozenset(("Y")))
    s = 0.428571
    c = 0.6
    l = 1.05
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset("X"))
    assert (recommend_rules["consequents"][2] == frozenset(("Z")))
    s = 0.285714
    c = 0.4
    l = 1.40
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("X"))
    assert (recommend_rules["consequents"][3] == frozenset(("A")))
    s = 0.142857
    c = 0.2
    l = 1.40
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)

    ## 6
    assert (recommend_rules["rank"][5] == 6)

    # items size is one, metric is "confidence", allow_from_items is True
    recommender = Recommender(association_rules)

    items = ["X"]
    recommend_rules = recommender.recommend(items,
                                            n=6,
                                            metric="confidence",
                                            allow_from_items=True)

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset("X"))
    assert (recommend_rules["consequents"][0] == frozenset(("B")))
    s = 0.428571
    c = 0.6
    l = 1.40
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset("X"))
    assert (recommend_rules["consequents"][1] == frozenset(("Y")))
    s = 0.428571
    c = 0.6
    l = 1.05
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset("X"))
    assert (recommend_rules["consequents"][2] == frozenset(("Z")))
    s = 0.285714
    c = 0.4
    l = 1.40
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("X"))
    assert (recommend_rules["consequents"][3] == frozenset(("A")))
    s = 0.142857
    c = 0.2
    l = 1.40
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)

    ## 6
    assert (recommend_rules["rank"][5] == 6)

    # items size is two or more, metric is "confidence", allow_from_items is False
    recommender = Recommender(association_rules)

    items = ["X", "Y"]
    recommend_rules = recommender.recommend(items, n=6, metric="confidence")

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][0] == frozenset(("Z")))
    s = 0.285714
    c = 0.666667
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset("X"))
    assert (recommend_rules["consequents"][1] == frozenset(("B")))
    s = 0.428571
    c = 0.600000
    l = 1.400000
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset("Y"))
    assert (recommend_rules["consequents"][2] == frozenset(("A")))
    s = 0.285714
    c = 0.500000
    l = 1.166667
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("Y"))
    assert (recommend_rules["consequents"][3] == frozenset(("C")))
    s = 0.142857
    c = 0.250000
    l = 0.875000
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)

    ## 6
    assert (recommend_rules["rank"][5] == 6)

    # items size is two or more, metric is "confidence", allow_from_items is True
    recommender = Recommender(association_rules)

    items = ["X", "Y"]
    recommend_rules = recommender.recommend(items,
                                            n=6,
                                            metric="confidence",
                                            allow_from_items=True)

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset(("Y")))
    assert (recommend_rules["consequents"][0] == frozenset(("X")))
    s = 0.428571
    c = 0.750000
    l = 1.050000
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][1] == frozenset(("Z")))
    s = 0.285714
    c = 0.666667
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset("X"))
    assert (recommend_rules["consequents"][2] == frozenset(("B")))
    s = 0.428571
    c = 0.600000
    l = 1.400000
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("X"))
    assert (recommend_rules["consequents"][3] == frozenset(("Y")))
    s = 0.428571
    c = 0.600000
    l = 1.050000
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)
    assert (recommend_rules["antecedents"][4] == frozenset("Y"))
    assert (recommend_rules["consequents"][4] == frozenset(("A")))
    s = 0.285714
    c = 0.500000
    l = 1.166667
    assert (s - 0.000001 <= recommend_rules["support"][4] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][4] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][4] <= l + 0.000001)

    ## 6
    assert (recommend_rules["rank"][5] == 6)
    assert (recommend_rules["antecedents"][5] == frozenset("Y"))
    assert (recommend_rules["consequents"][5] == frozenset(("C")))
    s = 0.142857
    c = 0.250000
    l = 0.875000
    assert (s - 0.000001 <= recommend_rules["support"][5] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][5] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][5] <= l + 0.000001)

    # items size is one, metric is "lift", allow_from_items is False
    recommender = Recommender(association_rules)

    items = ["X"]
    recommend_rules = recommender.recommend(items, n=6, metric="lift")

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset(("X")))
    assert (recommend_rules["consequents"][0] == frozenset(("B")))
    s = 0.428571
    c = 0.6
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset(("X")))
    assert (recommend_rules["consequents"][1] == frozenset(("Y")))
    s = 0.285714
    c = 0.4
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset("X"))
    assert (recommend_rules["consequents"][2] == frozenset(("Z")))
    s = 0.285714
    c = 0.4
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("X"))
    assert (recommend_rules["consequents"][3] == frozenset(("A")))
    s = 0.142857
    c = 0.2
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)

    ## 6
    assert (recommend_rules["rank"][5] == 6)

    # items size is one, metric is "lift", allow_from_items is True
    recommender = Recommender(association_rules)

    items = ["X"]
    recommend_rules = recommender.recommend(items,
                                            n=6,
                                            metric="lift",
                                            allow_from_items=True)

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset(("X")))
    assert (recommend_rules["consequents"][0] == frozenset(("B")))
    s = 0.428571
    c = 0.6
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset(("X")))
    assert (recommend_rules["consequents"][1] == frozenset(("Y")))
    s = 0.285714
    c = 0.4
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset("X"))
    assert (recommend_rules["consequents"][2] == frozenset(("Z")))
    s = 0.285714
    c = 0.4
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("X"))
    assert (recommend_rules["consequents"][3] == frozenset(("A")))
    s = 0.142857
    c = 0.2
    l = 1.4
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)

    ## 6
    assert (recommend_rules["rank"][5] == 6)

    # items size is two or more, metric is "lift", allow_from_items is False
    recommender = Recommender(association_rules)

    items = ["X", "Y"]
    recommend_rules = recommender.recommend(items, n=6, metric="lift")

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][0] == frozenset(("Z")))
    s = 0.285714
    c = 0.666667
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][1] == frozenset(("B")))
    s = 0.142857
    c = 0.333333
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][2] == frozenset(("A")))
    s = 0.142857
    c = 0.333333
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("Y"))
    assert (recommend_rules["consequents"][3] == frozenset(("C")))
    s = 0.142857
    c = 0.250000
    l = 0.875000
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)

    ## 6
    assert (recommend_rules["rank"][5] == 6)

    # items size is two or more, metric is "lift", allow_from_items is True
    recommender = Recommender(association_rules)

    items = ["X", "Y"]
    recommend_rules = recommender.recommend(items,
                                            n=6,
                                            metric="lift",
                                            allow_from_items=True)

    ## 1
    assert (recommend_rules["rank"][0] == 1)
    assert (recommend_rules["antecedents"][0] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][0] == frozenset(("Z")))
    s = 0.285714
    c = 0.666667
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][0] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][0] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][0] <= l + 0.000001)

    ## 2
    assert (recommend_rules["rank"][1] == 2)
    assert (recommend_rules["antecedents"][1] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][1] == frozenset(("B")))
    s = 0.142857
    c = 0.333333
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][1] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][1] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][1] <= l + 0.000001)

    ## 3
    assert (recommend_rules["rank"][2] == 3)
    assert (recommend_rules["antecedents"][2] == frozenset(("X", "Y")))
    assert (recommend_rules["consequents"][2] == frozenset(("A")))
    s = 0.142857
    c = 0.333333
    l = 2.333333
    assert (s - 0.000001 <= recommend_rules["support"][2] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][2] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][2] <= l + 0.000001)

    ## 4
    assert (recommend_rules["rank"][3] == 4)
    assert (recommend_rules["antecedents"][3] == frozenset("Y"))
    assert (recommend_rules["consequents"][3] == frozenset(("X")))
    s = 0.285714
    c = 0.500000
    l = 1.750000
    assert (s - 0.000001 <= recommend_rules["support"][3] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][3] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][3] <= l + 0.000001)

    ## 5
    assert (recommend_rules["rank"][4] == 5)
    assert (recommend_rules["antecedents"][4] == frozenset("X"))
    assert (recommend_rules["consequents"][4] == frozenset(("Y")))
    s = 0.285714
    c = 0.400000
    l = 1.400000
    assert (s - 0.000001 <= recommend_rules["support"][4] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][4] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][4] <= l + 0.000001)

    ## 6
    assert (recommend_rules["rank"][5] == 6)
    assert (recommend_rules["antecedents"][5] == frozenset("Y"))
    assert (recommend_rules["consequents"][5] == frozenset(("C")))
    s = 0.142857
    c = 0.250000
    l = 0.875000
    assert (s - 0.000001 <= recommend_rules["support"][5] <= s + 0.000001)
    assert (c - 0.000001 <= recommend_rules["confidence"][5] <= c + 0.000001)
    assert (l - 0.000001 <= recommend_rules["lift"][5] <= l + 0.000001)
