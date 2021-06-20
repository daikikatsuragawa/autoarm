import pandas as pd
from mlxtend.frequent_patterns import association_rules


class AssociationRules:
    """
    AssociationRules contains association rules.
    """
    __RANK = "rank"
    __ANTECEDENTS = "antecedents"
    __CONSEQUENTS = "consequents"
    __SUPPORT = "support"
    __CONFIDENCE = "confidence"
    __LIFT = "lift"
    __COLUMNS = [__ANTECEDENTS, __CONSEQUENTS, __SUPPORT, __CONFIDENCE, __LIFT]
    __SORT_BY_ANTECEDENTS_CONSEQUENTS = [__ANTECEDENTS, __CONSEQUENTS]
    __SORT_BY_CONFIDENCE_SUPPORT_LIFT = [__CONFIDENCE, __SUPPORT, __LIFT]
    __SORT_BY_LIFT_SUPPORT_CONFIDENCE = [__LIFT, __SUPPORT, __CONFIDENCE]
    __DEFAULT_SORT_BY_COLUMNS = {
        __CONFIDENCE: __SORT_BY_CONFIDENCE_SUPPORT_LIFT,
        __LIFT: __SORT_BY_LIFT_SUPPORT_CONFIDENCE,
    }
    __df = pd.DataFrame()

    def __init__(self, frequent_itemsets, metric=None, min_threshold=0.8):
        if metric is None:
            metric = self.__CONFIDENCE
        if self.__CONFIDENCE.__eq__(metric):
            if not 0 <= min_threshold <= 1:
                raise ValueError()
        elif self.__LIFT.__eq__(metric):
            if not 0 <= min_threshold:
                raise ValueError()
        else:
            raise ValueError()
        sort_by_columns = self.__DEFAULT_SORT_BY_COLUMNS[metric]
        association_rules_df = association_rules(
            frequent_itemsets.to_frame(), metric=metric, min_threshold=min_threshold
        )
        tmp_list = list()
        while len(sort_by_columns) > len(tmp_list):
            tmp_list.append(False)
        tmp_df = association_rules_df.sort_values(
            sort_by_columns, ascending=tmp_list
        ).reset_index(drop=True)
        self.__df = tmp_df[self.__COLUMNS]

    def to_frame(self):
        """
        Returns pandas.DataFrame.
        """
        return self.__df
