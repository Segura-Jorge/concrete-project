import pandas

def prepare_data1(df):
    """
    Convert the units from metric to imperial and add an index column 'sample'.
    
    Parameters:
    df (DataFrame): The original dataframe with metric units.
    
    Returns:
    DataFrame: The dataframe with units converted to imperial and an added 'sample' column.
    """
    # Conversion factors
    KG_TO_LBS = 2.20462
    MPA_TO_PSI = 145.038

    # Convert units to imperial
    df_imperial = df.copy()
    df_imperial['cement'] *= KG_TO_LBS
    df_imperial['slag'] *= KG_TO_LBS
    df_imperial['ash'] *= KG_TO_LBS
    df_imperial['water'] *= KG_TO_LBS
    df_imperial['superplastic'] *= KG_TO_LBS
    df_imperial['coarseagg'] *= KG_TO_LBS
    df_imperial['fineagg'] *= KG_TO_LBS
    df_imperial['strength'] *= MPA_TO_PSI

    return df_imperial


def prepare_data2(df):
    """
    Convert the units from metric to imperial and set the 'sample' column as the index.
    
    Parameters:
    df (DataFrame): The original dataframe with metric units.
    
    Returns:
    DataFrame: The dataframe with units converted to imperial and the 'sample' column set as the index.
    """
    # Conversion factors
    KG_TO_LBS = 2.20462
    MPA_TO_PSI = 145.038

    # Convert units to imperial
    # For the components measured in kg/m^3, first convert kg to lbs
    df[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg']] *= KG_TO_LBS
    # Convert compressive strength from MPa to psi
    df['strength'] *= MPA_TO_PSI

    # Add an index column named 'sample'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sample'}, inplace=True)
    
    # Set the 'sample' column as the index
    df.set_index('sample', inplace=True)
    
    return df



def prepare_data3(df):
    """
    Convert the units from metric to imperial and set the 'sample' column as the index.
    
    Parameters:
    df (DataFrame): The original dataframe with metric units.
    
    Returns:
    DataFrame: The dataframe with units converted to imperial and the 'sample' column set as the index.
    """
    # Conversion factors
    KG_TO_LBS = 2.20462
    MPA_TO_PSI = 145.038

    # Convert units to imperial
    # For the components measured in kg/m^3, first convert kg to lbs
    df[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg']] *= KG_TO_LBS
  
    # Convert compressive strength from MPa to psi
    df['strength'] *= MPA_TO_PSI

    # Add an index column named 'sample'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sample'}, inplace=True)
    
    # Set the 'sample' column as the index
    df.set_index('sample', inplace=True)
    
    # List of columns to sum
    components = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg']

    # Sum the values across the specified columns and create the new column
    df['total_lbs_per_yd^3'] = round(df[list(components)].sum(axis=1))

    return df


def prepared_concrete_data(df):
    """
    Convert the units from metric to imperial and set the 'sample' column as the index.
    
    Parameters:
    df (DataFrame): The original dataframe with metric units.
    
    Returns:
    DataFrame: The dataframe with units converted to imperial and the 'sample' column set as the index.
    """
    # Conversion factors
    KG_TO_LBS = 2.20462
    M3_TO_CUBIC_YARD = 1.30795
    MPA_TO_PSI = 145.038

    # Convert units to imperial
    # For the components measured in kg/m^3, first convert kg to lbs
    df[['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg']] *= KG_TO_LBS
    # Convert compressive strength from MPa to psi
    df['strength'] *= MPA_TO_PSI

    # Add an index column named 'sample'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sample'}, inplace=True)
    
    # Set the 'sample' column as the index
    df.set_index('sample', inplace=True)
    
    # List of components
    components = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg']

    # Sum the values across the specified columns and create the new column
    df['total_lbs_per_yd^3'] = round(df[list(components)].sum(axis=1))

    return df