import seaborn as sb
import xarray as xr
from matplotlib import pyplot as plt
import pickle

dataset = pickle.load( open( "../remote_data/data/22-04-2020_04-09/mse.pkl", "rb" ) )

dataset.plot( x="r_0", y="r_1", col="exponent", col_wrap=3 )
