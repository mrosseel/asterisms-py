# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_tycho2_dat_to_parquet.ipynb.

# %% auto 0
__all__ = ['read_tycho2']

# %% ../nbs/00_tycho2_dat_to_parquet.ipynb 7
def read_tycho2(filename):
    labels = [
        "TYC123", "pflag", "RAmdeg", "DEmdeg", "pmRA", "pmDE", 
        "e_RAmdeg", "e_DEmdeg", "e_pmRA", "e_pmDE", "EpRAm", "EpDEm", 
        "Num", "q_RAmdeg", "q_DEmdeg", "q_pmRA", "q_pmDE", "BTmag", 
        "e_BTmag", "VTmag", "e_VTmag", "prox", "TYC", "HIPCCDM", 
        "RAdeg", "DEdeg", "EpRA-1990", "EpDE-1990", "e_RAdeg", "e_DEdeg", 
        "posflg", "corr"
    ]
    
    df = pl.read_csv(filename, separator='|', has_header=False, new_columns=labels, 
                     dtypes={'RAmdeg': pl.Float32, 'DEmdeg': pl.Float32, 'BTmag': pl.Float32, 'e_BTmag': pl.Float32, 'VTmag': pl.Float32,'e_VTmag': pl.Float32, 'HIPCCDM': pl.Utf8})
    return df