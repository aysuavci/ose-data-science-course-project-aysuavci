"""This module contains auxiliary functions for RD predictions used in the main notebook."""

import numpy as np
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm_api
import matplotlib as pl
import matplotlib.pyplot as plt
from IPython.display import HTML
from stargazer.stargazer import Stargazer, LineLocation
from statsmodels.iolib.summary2 import summary_col

from auxiliary.auxiliary_tools import *
from auxiliary.auxiliary_plots import *
from auxiliary.auxiliary_tables import *

def get_Bayesian_predictions(df):
    
    df["post_1"] = df["prior_1"]/100

    df.loc[df['pos_comparisons'] == 0, 'post_1'] = 0
    df.loc[df['pos_comparisons'] == 1, 'post_1'] = 0
    df.loc[df['pos_comparisons'] == 2, 'post_1'] = 0
    df.loc[df['pos_comparisons'] == 3, 'post_1'] = df["prior_1"]/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_2"] = df["prior_2"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_2'] = (df["prior_2"]*(1/9)*(1/9)*(1/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_2'] = (df["prior_2"]*(24/9)*(1/9)*(1/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_2'] = (df["prior_2"]*(24/9)*(8/9)*(1/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_2'] = (df["prior_2"]*(8/9)*(8/9)*(8/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_3"] = df["prior_3"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_3'] = (df["prior_3"]*(2/9)*(2/9)*(2/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_3'] = (df["prior_3"]*(21/9)*(2/9)*(2/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_3'] = (df["prior_3"]*(21/9)*(7/9)*(2/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_3'] = (df["prior_3"]*(7/9)*(7/9)*(7/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_4"] = df["prior_4"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_4'] = (df["prior_4"]*(3/9)*(3/9)*(3/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_4'] = (df["prior_4"]*(18/9)*(3/9)*(3/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_4'] = (df["prior_4"]*(18/9)*(6/9)*(3/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_4'] = (df["prior_4"]*(6/9)*(6/9)*(6/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_5"] = df["prior_5"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_5'] = (df["prior_5"]*(4/9)*(4/9)*(4/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_5'] = (df["prior_5"]*(15/9)*(4/9)*(4/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_5'] = (df["prior_5"]*(15/9)*(5/9)*(4/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_5'] = (df["prior_5"]*(5/9)*(5/9)*(5/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_6"] = df["prior_6"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_6'] = (df["prior_6"]*(5/9)*(5/9)*(5/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_6'] = (df["prior_6"]*(12/9)*(5/9)*(5/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_6'] = (df["prior_6"]*(12/9)*(4/9)*(5/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_6'] = (df["prior_6"]*(4/9)*(4/9)*(4/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_7"] = df["prior_7"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_7'] = (df["prior_7"]*(6/9)*(6/9)*(6/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_7'] = (df["prior_7"]*(9/9)*(6/9)*(6/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_7'] = (df["prior_7"]*(9/9)*(3/9)*(6/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_7'] = (df["prior_7"]*(3/9)*(3/9)*(3/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_8"] = df["prior_8"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_8'] = (df["prior_8"]*(7/9)*(7/9)*(7/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_8'] = (df["prior_8"]*(6/9)*(7/9)*(7/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_8'] = (df["prior_8"]*(6/9)*(2/9)*(7/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_8'] = (df["prior_8"]*(2/9)*(2/9)*(2/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_9"] = df["prior_9"]/100 

    df.loc[df['pos_comparisons'] == 0, 'post_9'] = (df["prior_9"]*(8/9)*(8/9)*(8/9))/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9)+ df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_9'] = (df["prior_9"]*(3/9)*(8/9)*(8/9))/(df["prior_2"]*(24/9)*(1/9)*(1/9) + df["prior_3"]*(21/9)*(2/9)*(2/9) + df["prior_4"]*(18/9)*(3/9)*(3/9) + df["prior_5"]*(15/9)*(4/9)*(4/9) + df["prior_6"]*(12/9)*(5/9)*(5/9) + df["prior_7"]*(9/9)*(6/9)*(6/9) + df["prior_8"]*(6/9)*(7/9)*(7/9) + df["prior_9"]*(3/9)*(8/9)*(8/9))
    df.loc[df['pos_comparisons'] == 2, 'post_9'] = (df["prior_9"]*(3/9)*(1/9)*(8/9))/(df["prior_2"]*(24/9)*(8/9)*(1/9) + df["prior_3"]*(21/9)*(7/9)*(2/9) + df["prior_4"]*(18/9)*(6/9)*(3/9) + df["prior_5"]*(15/9)*(5/9)*(4/9) + df["prior_6"]*(12/9)*(4/9)*(5/9) + df["prior_7"]*(9/9)*(3/9)*(6/9) + df["prior_8"]*(6/9)*(2/9)*(7/9) + df["prior_9"]*(3/9)*(1/9)*(8/9))
    df.loc[df['pos_comparisons'] == 3, 'post_9'] = (df["prior_9"]*(1/9)*(1/9)*(1/9))/(df["prior_1"] + df["prior_2"]*(8/9)*(8/9)*(8/9) + df["prior_3"]*(7/9)*(7/9)*(7/9) + df["prior_4"]*(6/9)*(6/9)*(6/9) + df["prior_5"]*(5/9)*(5/9)*(5/9) + df["prior_6"]*(4/9)*(4/9)*(4/9) + df["prior_7"]*(3/9)*(3/9)*(3/9) + df["prior_8"]*(2/9)*(2/9)*(2/9) + df["prior_9"]*(1/9)*(1/9)*(1/9))

    df["post_10"] = df["prior_10"]/100

    df.loc[df['pos_comparisons'] == 0, 'post_10'] = df["prior_10"]/(df["prior_2"]*(1/9)*(1/9)*(1/9) + df["prior_3"]*(2/9)*(2/9)*(2/9) + df["prior_4"]*(3/9)*(3/9)*(3/9) + df["prior_5"]*(4/9)*(4/9)*(4/9) + df["prior_6"]*(5/9)*(5/9)*(5/9) + df["prior_7"]*(6/9)*(6/9)*(6/9) + df["prior_8"]*(7/9)*(7/9)*(7/9) + df["prior_9"]*(8/9)*(8/9)*(8/9) + df["prior_10"])
    df.loc[df['pos_comparisons'] == 1, 'post_10'] = 0
    df.loc[df['pos_comparisons'] == 2, 'post_10'] = 0
    df.loc[df['pos_comparisons'] == 3, 'post_10'] = 0    
    
    return 




def get_rank_dummies(df):
    df["rankdummy1"] = 0
    df.loc[df['rank'] == 1, 'rankdummy1'] = 1
    
    df["rankdummy2"] = 0
    df.loc[df['rank'] == 2, 'rankdummy2'] = 1
    
    df["rankdummy3"] = 0
    df.loc[df['rank'] == 3, 'rankdummy3'] = 1
    
    df["rankdummy4"] = 0
    df.loc[df['rank'] == 4, 'rankdummy4'] = 1
    
    df["rankdummy5"] = 0
    df.loc[df['rank'] == 5, 'rankdummy5'] = 1
    
    df["rankdummy6"] = 0
    df.loc[df['rank'] == 6, 'rankdummy6'] = 1
    
    df["rankdummy7"] = 0
    df.loc[df['rank'] == 7, 'rankdummy7'] = 1
    
    df["rankdummy8"] = 0
    df.loc[df['rank'] == 8, 'rankdummy8'] = 1
    
    df["rankdummy9"] = 0
    df.loc[df['rank'] == 9, 'rankdummy9'] = 1
    
    return 

def get_rank_interation_term(df):
    df["rankdummy1_interact"] = df["rankdummy1"] * df["dummytreat_direct1month"]
    df["rankdummy2_interact"] = df["rankdummy2"] * df["dummytreat_direct1month"]
    df["rankdummy3_interact"] = df["rankdummy3"] * df["dummytreat_direct1month"]
    df["rankdummy4_interact"] = df["rankdummy4"] * df["dummytreat_direct1month"]
    df["rankdummy5_interact"] = df["rankdummy5"] * df["dummytreat_direct1month"]
    df["rankdummy6_interact"] = df["rankdummy6"] * df["dummytreat_direct1month"]
    df["rankdummy7_interact"] = df["rankdummy7"] * df["dummytreat_direct1month"]
    df["rankdummy8_interact"] = df["rankdummy8"] * df["dummytreat_direct1month"]
    df["rankdummy9_interact"] = df["rankdummy9"] * df["dummytreat_direct1month"]
    
    return

def cluster_fit(formula, data, group_var):
    """
    To run regressions with standard errors clustered at subject level
    
    """
    fit = sm_api.OLS.from_formula(formula, data=data).fit()
    to_keep = pd.RangeIndex(len(data)).difference(pd.Index(fit.model.data.missing_row_idx))
    robust = fit.get_robustcov_results(cov_type='cluster',
                                       groups=data.iloc[to_keep][group_var])
    return robust

def get_variables_for_extension(df_control):
    #Dummies for analysis
    df_control['secondwave'] = np.nan
    df_control.loc[df_control['time'] == 2, 'secondwave'] = 1
    df_control.loc[df_control['time'] == 1, 'secondwave'] = 0
    df_control['dummy_feedback'] = np.nan
    df_control.loc[df_control['treatment'] == 'confidence_1monthlater', 'dummy_feedback'] = 1
    df_control.loc[df_control['treatment'] == 'nofeedback', 'dummy_feedback'] = 0
    #Interaction term
    df_control['feedback_secondwave'] = df_control['secondwave']*df_control['dummy_feedback']
    return 