import pandas as pd
import pytest

from autoarm import Dataset

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
    dataset = Dataset(
        sample_df, transaction_column=transaction_column, item_column=item_column
    )
    assert type(dataset.to_frame()) == pd.core.frame.DataFrame

    with pytest.raises(ValueError):
        dataset = Dataset(
            sample_df, transaction_column="other", item_column=item_column
        )

    with pytest.raises(ValueError):
        dataset = Dataset(
            sample_df, transaction_column=transaction_column, item_column="other"
        )
