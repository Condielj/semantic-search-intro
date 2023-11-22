import pandas as pd


def split_row(row):
    codes = [code.strip() for code in row.split(",") if code.strip()]
    return pd.Series(codes)


def format_input_df(df: pd.DataFrame, split_column: str = "hs_code") -> pd.DataFrame:
    # Apply the function to the 'hs_code' column and stack the results into a new DataFrame
    codes_df = (
        df[split_column]
        .apply(split_row)
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame(split_column)
    )

    # Duplicate the other columns based on the number of codes
    replicated_df = (
        df.loc[codes_df.index].reset_index(drop=True).drop(split_column, axis=1)
    )

    # Concatenate the replicated DataFrame with the 'hs_code' DataFrame
    new_df = pd.concat([replicated_df, codes_df], axis=1)

    return new_df
