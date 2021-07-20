"""This module contains auxiliary functions for the creation of tables in the main notebook."""

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

from auxiliary.example_project_auxiliary_predictions import *
from auxiliary.example_project_auxiliary_plots import *
from auxiliary.auxiliary_tables import *

def Main_Table_1(df):
    
    df_good = pd.DataFrame({"beliefadjustment_normalized": df[df["dummynews_goodbad"] == 0]['beliefadjustment_normalized'], "dummytreat_direct1month": df[df["dummynews_goodbad"] == 0]['dummytreat_direct1month'], "rank": df[df["dummynews_goodbad"] == 0]['rank'], "beliefadjustment_bayes_norm": df[df["dummynews_goodbad"] == 0]['beliefadjustment_bayes_norm']})
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month", data=df_good)
    reg_1 = model_ols.fit(cov_type='HC1')
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month + rank + beliefadjustment_bayes_norm", data=df_good)
    reg_2 = model_ols.fit(cov_type='HC1')
    
    df_bad = pd.DataFrame({"beliefadjustment_normalized": df[df["dummynews_goodbad"] == 1]['beliefadjustment_normalized'], "dummytreat_direct1month": df[df["dummynews_goodbad"] == 1]['dummytreat_direct1month'], "rank": df[df["dummynews_goodbad"] == 1]['rank'], "beliefadjustment_bayes_norm": df[df["dummynews_goodbad"] == 1]['beliefadjustment_bayes_norm']})
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month", data=df_bad)
    reg_3 = model_ols.fit(cov_type='HC1')
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month + rank + beliefadjustment_bayes_norm", data=df_bad)
    reg_4 = model_ols.fit(cov_type='HC1')
    
    #Generating interaction term
    df["interact_direct1month"] = df["dummytreat_direct1month"]*df["dummynews_goodbad"]

    model_ols = smf.ols(formula= "beliefadjustment_normalized ~ dummytreat_direct1month + dummynews_goodbad + interact_direct1month", data=df)
    reg_5 = model_ols.fit(cov_type='HC1')
    model_ols = smf.ols(formula= "beliefadjustment_normalized ~ dummytreat_direct1month + dummynews_goodbad + rank + interact_direct1month + beliefadjustment_bayes_norm", data=df)
    reg_6 = model_ols.fit(cov_type='HC1')
    
    model_ols = smf.ols(formula= "beliefadjustment_normalized ~ dummytreat_direct1month + dummynews_goodbad + interact_direct1month + rankdummy1 + rankdummy2 + rankdummy3 + rankdummy4 + rankdummy5 + rankdummy6 + rankdummy7 + rankdummy8 + rankdummy9 + rankdummy1_interact + rankdummy2_interact + rankdummy3_interact + rankdummy4_interact + rankdummy5_interact + rankdummy6_interact + rankdummy7_interact + rankdummy8_interact + rankdummy9_interact", data=df)
    reg_7 = model_ols.fit(cov_type='HC1')
    model_ols = smf.ols(formula= "beliefadjustment_normalized ~ dummytreat_direct1month + dummynews_goodbad + interact_direct1month + beliefadjustment_bayes_norm + rankdummy1 + rankdummy2 + rankdummy3 + rankdummy4 + rankdummy5 + rankdummy6 + rankdummy7 + rankdummy8 + rankdummy9 + rankdummy1_interact + rankdummy2_interact + rankdummy3_interact + rankdummy4_interact + rankdummy5_interact + rankdummy6_interact + rankdummy7_interact + rankdummy8_interact + rankdummy9_interact", data=df)
    reg_8 = model_ols.fit(cov_type='HC1')
    
    Main_Table_1 = Stargazer([reg_1, reg_2, reg_3, reg_4, reg_5, reg_6, reg_7, reg_8])
    Main_Table_1.title('Table 1 - Belief Adjustment: Direct versus One Month Later')
    Main_Table_1.dependent_variable_name('Normalized Belief Adjustment - ')
    Main_Table_1.custom_columns(['Positive Information', 'Negative Information', 'Difference-in-difference', 'Difference-in-difference with rank fixed effects'], [2,2,2,2])
    
    return Main_Table_1
   
    
def Appendix_Table_1(df):
    df_good = pd.DataFrame({"beliefadjustment_normalized": df[df["dummynews_goodbad_h"] == 0]['beliefadjustment_normalized'],
                            "dummytreat_direct1month": df[df["dummynews_goodbad_h"] == 0]['dummytreat_direct1month'], 
                            "rank": df[df["dummynews_goodbad_h"] == 0]['rank'],
                            "beliefadjustment_bayes_norm": df[df["dummynews_goodbad_h"] == 0]['beliefadjustment_bayes_norm']})
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month", data=df_good)
    reg_1 = model_ols.fit(cov_type='HC1')
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month + rank + beliefadjustment_bayes_norm", data=df_good)
    reg_2 = model_ols.fit(cov_type='HC1')
    df_bad = pd.DataFrame({"beliefadjustment_normalized": df[df["dummynews_goodbad_h"] == 1]['beliefadjustment_normalized'],
                           "dummytreat_direct1month": df[df["dummynews_goodbad_h"] == 1]['dummytreat_direct1month'],
                           "rank": df[df["dummynews_goodbad_h"] == 1]['rank'],
                           "beliefadjustment_bayes_norm": df[df["dummynews_goodbad_h"] == 1]['beliefadjustment_bayes_norm']})
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month", data=df_bad)
    reg_3 = model_ols.fit(cov_type='HC1')
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ dummytreat_direct1month + rank + beliefadjustment_bayes_norm",
                        data=df_bad)
    reg_4 = model_ols.fit(cov_type='HC1')

    #Generating interaction term
    df["interact_direct1month"] = df["dummytreat_direct1month"]*df["dummynews_goodbad"]
    
    model_ols = smf.ols(formula= 
                        "beliefadjustment_normalized ~ dummytreat_direct1month + dummynews_goodbad_h + interact_direct1month",
                        data=df)
    reg_5 = model_ols.fit(cov_type='HC1')
    model_ols = smf.ols(formula= 
                        "beliefadjustment_normalized ~ dummytreat_direct1month + dummynews_goodbad_h + rank + interact_direct1month + beliefadjustment_bayes_norm", data=df)
    reg_6 = model_ols.fit(cov_type='HC1')

    Appendix_Table_1 = Stargazer([reg_1, reg_2, reg_3, reg_4, reg_5, reg_6])
    Appendix_Table_1.title('Appendix Table 1 - Belief Adjustment: Direct versus One Month Later')
    Appendix_Table_1.dependent_variable_name('Normalized Belief Adjustment - ')
    Appendix_Table_1.custom_columns(['Positive Information', 'Negative Information', 'Difference-in-difference'], [2,2,2])
    
    return Appendix_Table_1

def Appendix_Table_3(df):

    df_short_g = pd.DataFrame({"beliefadjustment_normalized": df[df["dummynews_goodbad"] == 0][df[df["dummynews_goodbad"] == 0]['treatgroup'] == 4]['beliefadjustment_normalized'], "beliefadjustment_bayes_norm": df[df["dummynews_goodbad"] == 0][df[df["dummynews_goodbad"] == 0]['treatgroup'] == 4]['beliefadjustment_bayes_norm']})
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ beliefadjustment_bayes_norm", data=df_short_g)
    reg_s_1 = model_ols.fit(cov_type='HC1')

    df_short_b = pd.DataFrame({"beliefadjustment_normalized": df[df["dummynews_goodbad"] == 1][df[df["dummynews_goodbad"] == 1]['treatgroup'] == 4]['beliefadjustment_normalized'], "beliefadjustment_bayes_norm": df[df["dummynews_goodbad"] == 1][df[df["dummynews_goodbad"] == 1]['treatgroup'] == 4]['beliefadjustment_bayes_norm']})
    model_ols = smf.ols(formula="beliefadjustment_normalized ~ beliefadjustment_bayes_norm", data=df_short_b)
    reg_s_2 = model_ols.fit(cov_type='HC1')

    df["interact_negative_bayes"] = df["beliefadjustment_bayes_norm"]*df["dummynews_goodbad"]

    model_ols = smf.ols(formula= "beliefadjustment_normalized ~ beliefadjustment_bayes_norm + dummynews_goodbad + interact_negative_bayes", data=df[df['treatgroup'] == 4])
    reg_s_3 = model_ols.fit(cov_type='HC1')

    Table8 = Stargazer([reg_s_1, reg_s_2, reg_s_3])
    Table8.title('Table 8: Belief Adjustment in the Short-Run')
    Table8.dependent_variable_name('Belief Adjustment')
    Table8.custom_columns(['Positive Information', 'Negative Information', 'Difference-in-difference'], [1,1,1])
    
    return Appendix_Table_3

def cluster_fit(formula, data, group_var):
    """
    To run regressions with standard errors clustered at subject level
    """
    fit = sm_api.OLS.from_formula(formula, data=data).fit()
    to_keep = pd.RangeIndex(len(data)).difference(pd.Index(fit.model.data.missing_row_idx))
    robust = fit.get_robustcov_results(cov_type='cluster',
                                       groups=data.iloc[to_keep][group_var])
    return robust

def Extension_Table_1(df_ex):
    reg_B = cluster_fit('meanbeliefimage ~ meanbelief_priorbayesimage + mb_fracup + frac_upimage',
                        data=df_ex[(((df_ex['frac_upimage'] == 0) | (df_ex['frac_upimage'] == 1)) & (df_ex['IQtask'] ==0))],
                        group_var='ID')
    reg_IQ = cluster_fit('meanbeliefimage ~ meanbelief_priorbayesimage + mb_fracup + frac_upimage',
                         data=df_ex[(((df_ex['frac_upimage'] == 0) | (df_ex['frac_upimage'] == 1)) & (df_ex['IQtask'] ==1))],
                        group_var='ID')
    reg_C = cluster_fit('meanbeliefcard ~ meanbelief_priorbayescard + mb_fracupcard + frac_upcard',
                        data=df_ex[(df_ex['frac_upcard'] == 0) | (df_ex['frac_upcard'] == 1)],
                        group_var='ID')
    return print(summary_col([reg_B,reg_IQ,reg_C],stars=True,float_format='%0.2f', model_names=['Beauty','IQ','Control']))