# AutoARM

AutoARM simplifies and automates association rule mining and related tasks. With just a few lines of code, you can format data and make recommendations based on the results of association rule mining.

If you use a rule-based recommender system, you are able to explain the rationale for the recommendation. For example, a recommendation such as "You should buy B, because 90% of those who bought A that you are already considering buying also buy B.". Then, the people who received the explanation are satisfied with the recommendation. As a result, it will encourage their actions.

## Overview of Recommender

1. Split rules consequents
2. Identify rules that match the items entered
3. Exclude rules that have entered items in consequents (This procces can be skipped)
4. Sort rules by confidence or lift specified as a metric
5. Exclude duplicate consequents and leave higher rules
6. Output rules of the specified size

## Example

Recommender that use association rules are generated as follows.

```python
import pandas as pd

from autoarm import AssociationRules, Dataset, FrequentItemsets, Recommender

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
recommender = Recommender(association_rules)
```

Recommender, which is provided with the necessary information such as items, outputs useful rules for recommendation.

```python
items = ["X", "Y"]
recommend_rules = recommender.recommend(items, n=3, metric="confidence")
recommend_rules
```

|rank|  antecedents   |  consequents   |support |confidence|  lift  |
|---:|----------------|----------------|-------:|---------:|-------:|
|   1|(X, Y)          |(Z)             |0.285714|  0.666667|2.333333|
|   2|(X)             |(B)             |0.428571|  0.600000|1.400000|
|   3|(Y)             |(A)             |0.285714|  0.500000|1.166667|

For example, based on this output, recommendations can be made using the following description. "You should buy Z, because 66% of those who bought X and Y that you are already considering buying also buy Z.".
