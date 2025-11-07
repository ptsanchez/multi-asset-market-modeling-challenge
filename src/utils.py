import pandas as pd

def align_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Align numerous dataframes on their timestamps and return merged dataframe.
    """

    from functools import reduce
    return reduce(lambda left, right: left.join(right, how='inner'), dfs)