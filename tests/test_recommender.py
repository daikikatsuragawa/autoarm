import copy

import numpy as np
import pandas as pd
from df4loop import DFIterator


class Recommender:
    """
    Recommender contains association rules for recommended use.
    """
    __RANK = "rank"
    __ANTECEDENTS = "antecedents"
    __CONSEQUENTS = "consequents"
    __SUPPORT = "support"
    __CONFIDENCE = "confidence"
    __LIFT = "lift"
    __COLUMNS = [__ANTECEDENTS, __CONSEQUENTS, __SUPPORT, __CONFIDENCE, __LIFT]
    __COLUMNS_WITH_RANK = [
        __RANK,
        __ANTECEDENTS,
        __CONSEQUENTS,
        __SUPPORT,
        __CONFIDENCE,
        __LIFT,
    ]
    __SORT_BY_ANTECEDENTS_CONSEQUENTS = [__ANTECEDENTS, __CONSEQUENTS]
    __SORT_BY_CONFIDENCE_SUPPORT_LIFT = [__CONFIDENCE, __SUPPORT, __LIFT]
    __SORT_BY_LIFT_SUPPORT_CONFIDENCE = [__LIFT, __SUPPORT, __CONFIDENCE]
    __DEFAULT_SORT_BY_COLUMNS = {
        __CONFIDENCE: __SORT_BY_CONFIDENCE_SUPPORT_LIFT,
        __LIFT: __SORT_BY_LIFT_SUPPORT_CONFIDENCE,
    }
    __DEFAULT_N = 1
    __divided_rules = pd.DataFrame()
    __n = 1
    __metric = "confidence"
    __allow_from_items = False

    def __init__(self, association_rules, n=None, metric=None, allow_from_items=False):
        self.__divided_rules = self.__divide_consequents(association_rules.to_frame())
        if n is None:
            n = self.__DEFAULT_N
        if n < 1:
            raise ValueError()
        if metric is None:
            metric = self.__CONFIDENCE
        if metric not in [self.__CONFIDENCE, self.__LIFT]:
            raise ValueError()
        self.__n = n
        self.__metric = metric
        self.__allow_from_items = allow_from_items

    def recommend(
        self, items, n=None, metric=None, allow_from_items=None, sort_by_columns=None
    ):
        """
        Returns association rules that are useful for recommendations.
        """
        if len(frozenset(items)) < 1:
            raise ValueError()
        if n is None:
            n = self.__n
        if n < 1:
            raise ValueError()
        if metric is None:
            metric = self.__metric
        if metric not in [self.__CONFIDENCE, self.__LIFT]:
            raise ValueError()
        if allow_from_items is None:
            allow_from_items = self.__allow_from_items
        rules_df = self.__match_with_input_items(self.__divided_rules, frozenset(items))
        if not allow_from_items:
            rules_df = self.__exclude_input_items(rules_df, frozenset(items))
        if sort_by_columns is None:
            sort_by_columns = list()
        if len(sort_by_columns).__eq__(0):
            sort_by_columns = self.__DEFAULT_SORT_BY_COLUMNS[metric]
        return self.__select_top_n_consequences(
            rules_df, n, sort_by_columns=sort_by_columns
        )

    def __divide_consequents(self, rules_df):
        """
        Divide the consequents.
        """
        if len(rules_df).__eq__(0):
            return pd.DataFrame()
        df_iterator = DFIterator(rules_df)
        tmp_dist = dict()
        for row in df_iterator.iterrows(return_indexes=False):
            for consequent in row[self.__CONSEQUENTS]:
                tmp_row = copy.deepcopy(row)
                tmp_row[self.__CONSEQUENTS] = frozenset(consequent)
                tmp_dist[len(tmp_dist)] = tmp_row
        divided_rules_df = pd.DataFrame.from_dict(tmp_dist, orient="index")
        if len(divided_rules_df).__eq__(0):
            return pd.DataFrame()
        divided_rules_df.columns = rules_df.columns
        return divided_rules_df

    def __match_with_input_items(self, rules_df, items):
        """
        Match rules with items.
        """
        if len(rules_df).__eq__(0):
            return pd.DataFrame()
        df_iterator = DFIterator(rules_df)
        tmp_dist = dict()
        for row in df_iterator.iterrows(return_indexes=False):
            if frozenset(items).issuperset(row[self.__ANTECEDENTS]):
                tmp_dist[len(tmp_dist)] = copy.deepcopy(row)
        matched_rules_df = pd.DataFrame.from_dict(tmp_dist, orient="index")
        if len(matched_rules_df).__eq__(0):
            return pd.DataFrame()
        matched_rules_df.columns = rules_df.columns
        return matched_rules_df

    def __exclude_input_items(self, rules_df, items):
        """
        Exclude input items from recommendation candidates.
        """
        if len(rules_df).__eq__(0):
            return pd.DataFrame()
        df_iterator = DFIterator(rules_df)
        tmp_dist = dict()
        for row in df_iterator.iterrows(return_indexes=False):
            if not frozenset(items).issuperset(row[self.__CONSEQUENTS]):
                tmp_dist[len(tmp_dist)] = copy.deepcopy(row)
        excluded_rules_df = pd.DataFrame.from_dict(tmp_dist, orient="index")
        if len(excluded_rules_df).__eq__(0):
            return pd.DataFrame()
        excluded_rules_df.columns = rules_df.columns
        return excluded_rules_df

    def __select_top_n_consequences(self, rules_df, n=None, sort_by_columns=None):
        """
        Select top n consequences. Sort based on the specified metric.
        """
        if len(rules_df).__eq__(0):
            return self.__make_n_rows(rules_df, n)
        rules_df = rules_df.sort_values(
            by=self.__SORT_BY_ANTECEDENTS_CONSEQUENTS
        ).reset_index(drop=True)
        if n is None:
            n = self.__DEFAULT_N
        if sort_by_columns is None:
            sort_by_columns = list()
        if len(sort_by_columns) > 0:
            tmp_list = list()
            while len(sort_by_columns) > len(tmp_list):
                tmp_list.append(False)
            rules_df = rules_df.sort_values(
                by=sort_by_columns, ascending=tmp_list
            ).reset_index(drop=True)
        df_iterator = DFIterator(rules_df)
        already_selected_consequents = set()
        tmp_dist = dict()
        for row in df_iterator.iterrows(return_indexes=False):
            if not already_selected_consequents.issuperset(row[self.__CONSEQUENTS]):
                tmp_dist[len(tmp_dist)] = copy.deepcopy(row)
                for consequent in row[self.__CONSEQUENTS]:
                    already_selected_consequents.add(consequent)
        tmp_df = pd.DataFrame.from_dict(tmp_dist, orient="index")
        tmp_df.columns = rules_df.columns
        tmp_df = self.__make_n_rows(tmp_df, n)
        selected_rules_df = tmp_df.reset_index(drop=True)
        selected_rules_df[self.__RANK] = range(1, n + 1)
        return selected_rules_df[self.__COLUMNS_WITH_RANK]

    def __make_n_rows(self, df, n):
        """
        Adjust the pandas.DataFrame to n rows. The value of the added rows are NaN.
        """
        while len(df) < n:
            df = df.append([np.nan])
        while len(df) > n:
            df = df[:-1]
        return df
