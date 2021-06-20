import pandas as pd
from mlxtend.frequent_patterns import apriori


class FrequentItemsets:
    """
    FrequentItemsets contains frequent itemsets.
    """
    __SUPPORT = "support"
    __df = pd.DataFrame()

    def __init__(self, dataset, min_support=0.5):
        if not 0 < min_support <= 1:
            raise ValueError()
        frequent_items_df = apriori(
            dataset.to_frame(), min_support=min_support, use_colnames=True
        )
        self.__df = frequent_items_df.sort_values(
            self.__SUPPORT, ascending=False
        ).reset_index(drop=True)

    def to_frame(self):
        """
        Returns pandas.DataFrame.
        """
        return self.__df
