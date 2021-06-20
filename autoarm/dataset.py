import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


class Dataset:
    """
    Dataset contains dataset and has been converted to a format suitable for later processes.
    """
    __df = pd.DataFrame()

    def __init__(self, df, transaction_column, item_column):
        if transaction_column not in df.columns.tolist():
            raise ValueError()
        if item_column not in df.columns.tolist():
            raise ValueError()
        tmp_df = df.groupby(transaction_column)[item_column].apply(list)
        transaction_encoder = TransactionEncoder()
        tmp_df2 = transaction_encoder.fit(tmp_df).transform(tmp_df)
        self.__df = pd.DataFrame(tmp_df2, columns=transaction_encoder.columns_)

    def to_frame(self):
        """
        Returns pandas.DataFrame.
        """
        return self.__df
