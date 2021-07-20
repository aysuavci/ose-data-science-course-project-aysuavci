"""This module contains auxiliary functions for plotting which are used in the main notebook."""

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

def Main_Figure1(df):
    #Limit the values at +-50
    df_fig1 = df
    df_fig1.loc[(df['beliefadjustment'] > 50) & (df['beliefadjustment'] < 101), 'beliefadjustment'] = 50
    df_fig1.loc[(df['beliefadjustment'] < - 50) & (df['beliefadjustment'] > -101), 'beliefadjustment'] = -50

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig.suptitle('FIGURE 1', fontsize=15)
    #Direct & Positive
    ax1.hist(df_fig1[(df_fig1['treatgroup'] == 3) | (df_fig1['treatgroup'] == 4)][df_fig1[(df_fig1['treatgroup'] == 3) | (df_fig1['treatgroup'] == 4)]["dummynews_goodbad"] == 0]['beliefadjustment'],range=(-50, 50), bins=50)
    #Direct & Negative
    ax2.hist(df_fig1[(df_fig1['treatgroup'] == 3) | (df_fig1['treatgroup'] == 4)][df_fig1[(df_fig1['treatgroup'] == 3) | (df_fig1['treatgroup'] == 4)]["dummynews_goodbad"] == 1]['beliefadjustment'],range=(-50, 50), bins=50)
    
    #1-month & Positive
    ax3.hist(df_fig1[df_fig1['treatgroup'] == 2][df_fig1[df_fig1['treatgroup'] == 2]["dummynews_goodbad"] == 0]['beliefadjustment'],range=(-50, 50), bins=50)
    #1-month & Negative
    ax4.hist(df_fig1[df_fig1['treatgroup'] == 2][df_fig1[df_fig1['treatgroup'] == 2]["dummynews_goodbad"] == 1]['beliefadjustment'],range=(-50, 50), bins=50)
    
    ax1.set_title("Panel A. ConfidenceDirect: positive & negative")
    ax1.set_ylabel('Fraction')
    ax1.set_xlabel('Positive')
    ax2.set_xlabel('Negative')

    ax3.set_title("Panel B. Confidence1month: positive & negative")
    ax3.set_ylabel('Fraction')
    ax3.set_xlabel('Positive')
    ax4.set_xlabel('Negative')
    
    return Main_Figure1

def Main_Figure2(df):
    fig, ax = plt.subplots(2, figsize=(8, 8))
    
    #PANEL A
    ax[0].scatter(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av'], color='b', label='Prior')
    ax[0].plot(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av'], color='b')
    
    ax[0].scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r', label='Posterior Positive Feedback')
    ax[0].plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r')
    
    ax[0].scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g', label='Posterior Negative Feedback')
    ax[0].plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g')

    
    #PANEL B
    ax[1].scatter(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av'], color='b', label='Prior')
    ax[1].plot(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av'], color='b')
    ax[1].scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r', label='Posterior Positive Feedback')
    ax[1].plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r')
    ax[1].scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g', label='Posterior Negative Feedback')
    ax[1].plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g')

    ax[0].set_ylabel('Pr(upperhalf)')
    ax[0].set_title('Panel A. ConfidenceDirect')
    ax[0].legend(loc='lower right', fontsize = 'small')
    ax[0].set_ylim([30,90])
    fig.suptitle('FIGURE 2', fontsize=15)
    ax[1].legend(loc='lower right', fontsize = 'small')
    ax[1].set_xlabel('Test Performance')
    ax[1].set_ylabel('Pr(upperhalf)')
    ax[1].set_title('Panel B. Confidence1month')
    ax[1].set_ylim([30,90])
    
    return Main_Figure2

def Appendix_Figure_1(df):
    #censor at +/-50
    df_fig1 = df
    df_fig1.loc[(df['beliefadjustment'] > 50) & (df['beliefadjustment'] < 101), 'beliefadjustment'] = 50
    df_fig1.loc[(df['beliefadjustment'] < - 50) & (df['beliefadjustment'] > -101), 'beliefadjustment'] = -50

    df_fig_NF = df_fig1[df_fig1['treatgroup'] == 7]
    fig, Appendix_Figure_1 = plt.subplots(1, figsize=(5, 5))
    fig.suptitle('Appendix A.7 - No Feedback Condition', fontsize=15)
    Appendix_Figure_1.hist(df_fig_NF['beliefadjustment'],range=(-50, 50), bins=50)

    Appendix_Figure_1.set_title("No Feedback")
    Appendix_Figure_1.set_ylabel('Fraction')
    Appendix_Figure_1.set_xlabel('Belief Adjustment')
    
    return Appendix_Figure_1

def Appendix_Figure_2(df):
    fig, ax = plt.subplots(2, figsize=(8, 8))
    ax[0].scatter(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av_b'], color='b', label='Prior')
    ax[0].plot(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av_b'], color='b')
    ax[0].scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r', label='Posterior Bayes Positive Feedback')
    ax[0].plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r')
    ax[0].scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g', label='Posterior Bayes Negative Feedback')
    ax[0].plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g')

    ax[1].scatter(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av_b'], color='b', label='Prior')
    ax[1].plot(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av_b'], color='b')
    ax[1].scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r', label='Posterior Bayes Positive Feedback')
    ax[1].plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r')
    ax[1].scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g', label='Posterior Bayes Negative Feedback')
    ax[1].plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g')

    ax[0].set_ylabel('Pr(upperhalf)')
    ax[0].set_title('ConfidenceDirect')
    ax[0].legend(loc='lower right', fontsize = 'small')
    ax[0].set_ylim([10,100])
    fig.suptitle('Appendix A.8 - Figures Bayesian Posteriors', fontsize=15)
    ax[1].legend(loc='lower right', fontsize = 'small')
    ax[1].set_xlabel('Test Performance')
    ax[1].set_ylabel('Pr(upperhalf)')
    ax[1].set_title('Confidence1month')
    ax[1].set_ylim([0,100])
    
    return Appendix_Figure_2

def Extension_Figure_1(df):
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig.suptitle('EXTENSION - FIGURE 1: Bayesian & Observed Posterior Beliefs', fontsize=15)
    fig.suptitle('EXTENSION - FIGURE 1: Bayesian & Observed Posterior Beliefs', fontsize=15)
    
    """
    Basically, a combination of Main_Figure_2 and Appendix_Figure_2.
    
    """
    ax2.scatter(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av'], color='b', label='Prior')
    ax2.plot(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av'], color='b')
    ax2.scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r', label='Posterior Positive Feedback')
    ax2.plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r')
    ax2.scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g', label='Posterior Negative Feedback')
    ax2.plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g')

    ax4.scatter(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av'], color='b', label='Prior')
    ax4.plot(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av'], color='b')
    ax4.scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r', label='Posterior Positive Feedback')
    ax4.plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos'], color='r')
    ax4.scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g', label='Posterior Negative Feedback')
    ax4.plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg'], color='g')

    ax4.set_ylabel('Pr(upperhalf)', fontsize=10)
    ax4.legend(loc='lower right', fontsize = 'small')
    ax4.set_ylim([10,100])
    ax2.legend(loc='lower right', fontsize = 'small')
    ax2.set_xlabel('Test Performance')
    ax2.set_ylabel('Pr(upperhalf)', fontsize=10)
    ax2.set_ylim([10,100])
    ax2.set_title('Observed Posterior Belief', fontsize=14)

    ax1.scatter(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av_b'], color='b', label='Prior')
    ax1.plot(df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 0].sort_values(by=['test_1'])['prior_av_b'], color='b')
    ax1.scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r', label='Posterior Bayes Positive Feedback')
    ax1.plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r')
    ax1.scatter(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g', label='Posterior Bayes Negative Feedback')
    ax1.plot(df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 0) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g')

    ax3.scatter(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av_b'], color='b', label='Prior')
    ax3.plot(df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['test_1'], df[df['dummytreat_direct1month'] == 1].sort_values(by=['test_1'])['prior_av_b'], color='b')
    ax3.scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r', label='Posterior Bayes Positive Feedback')
    ax3.plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==0)].sort_values(by=['test_1'])['post_av_pos_b'], color='r')
    ax3.scatter(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g', label='Posterior Bayes Negative Feedback')
    ax3.plot(df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['test_1'], df[(df['dummytreat_direct1month'] == 1) & (df['dummynews_goodbad'] ==1)].sort_values(by=['test_1'])['post_av_neg_b'], color='g')

    ax1.set_ylabel('Confidence Direct', fontsize=14)
    ax1.set_title('Bayesian Posterior Beliefs', fontsize=14)
    ax1.legend(loc='lower right', fontsize = 'small')
    ax1.set_ylim([10,100])
    ax3.legend(loc='lower right', fontsize = 'small')
    ax3.set_xlabel('Test Performance')
    ax3.set_ylabel('Confidence 1-month', fontsize=14)
    ax3.set_ylim([10,100])
    
    return Extension_Figure_1
    
    
    
def Extension_Figure_2(df_ex):
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(20,5))
    fig.suptitle('FIGURE 2. Noise in Round-to-Round Updating by Treatment and Signal Type')

    sns.set_style('whitegrid')
    sns.regplot('meanbelief_priorbayesimage','meanbeliefimage',
                df_ex[(df_ex['frac_upimage'] == 0) & (df_ex['IQtask'] ==0) & (df_ex['round'] > 0)],
                scatter_kws={'s':30},line_kws={'color':'lightblue'}, marker="+", ax=axes[0], label='Bad news')
    sns.regplot('meanbelief_priorbayesimage','meanbeliefimage',
                df_ex[(df_ex['frac_upimage'] == 1) & (df_ex['IQtask'] ==0) & (df_ex['round'] > 0)],
                scatter_kws={'s':20},line_kws={'color':'orange'}, ax=axes[0],  label='Good news')

    sns.regplot('meanbelief_priorbayesimage','meanbeliefimage',
                df_ex[(df_ex['frac_upimage'] == 0) & (df_ex['IQtask'] ==1) & (df_ex['round'] > 0)],
                scatter_kws={'s':30},line_kws={'color':'lightblue'}, marker="+", ax=axes[1], label='Bad news')
    sns.regplot('meanbelief_priorbayesimage','meanbeliefimage',
                df_ex[(df_ex['frac_upimage'] == 1) & (df_ex['IQtask'] ==1) & (df_ex['round'] > 0)],
                scatter_kws={'s':20},line_kws={'color':'orange'}, ax=axes[1], label='Good news')

    sns.regplot('meanbelief_priorbayescard','meanbeliefcard',
                df_ex[(df_ex['frac_upcard'] == 0) & (df_ex['round'] > 0)],
                scatter_kws={'s':30},line_kws={'color':'lightblue'}, marker="+", ax=axes[2], label='Bad news')
    sns.regplot('meanbelief_priorbayescard','meanbeliefcard',
                df_ex[(df_ex['frac_upcard'] == 1) & (df_ex['round'] > 0)],
                scatter_kws={'s':20},line_kws={'color':'orange'}, ax=axes[2],  label='Good news')

    axes[0].set_title('Panel A. Beauty')
    axes[0].set_xlabel('Bayesian posterior mean')
    axes[0].set_ylabel('Posterior mean of Subjects')
    axes[0].legend(loc='lower right')
    axes[1].set_title('Panel B. IQ')
    axes[1].set_xlabel('Bayesian posterior mean')
    axes[1].set_ylabel('Posterior mean of Subjects')
    axes[1].legend(loc='lower right')
    axes[2].set_title('Panel C. Control')
    axes[2].set_xlabel('Bayesian posterior mean, using priors of subjects')
    axes[2].set_ylabel('Posterior mean of Subjects')
    axes[2].legend(loc='lower right')
    plt.show()
    return Extension_Figure_2


def cluster_fit(formula, data, group_var):
    """
    To run regressions with standard errors clustered at subject level
    """
    fit = sm_api.OLS.from_formula(formula, data=data).fit()
    to_keep = pd.RangeIndex(len(data)).difference(pd.Index(fit.model.data.missing_row_idx))
    robust = fit.get_robustcov_results(cov_type='cluster',
                                       groups=data.iloc[to_keep][group_var])
    return robust


def Extension_Figure_3(df_ex):
    #Regressions with clustered standard errors at subject level
    
    reg_B_b_ = cluster_fit('meanbeliefimage ~ meanbelief_priorbayesimage + mb_fracup + frac_upimage',
                           data=df_ex[(df_ex['frac_upimage'] == 0) & (df_ex['IQtask'] ==0)], group_var='ID')
    reg_B_g_ = cluster_fit('meanbeliefimage ~ meanbelief_priorbayesimage + mb_fracup + frac_upimage',
                           data=df_ex[(df_ex['frac_upimage'] == 1) & (df_ex['IQtask'] ==0)], group_var='ID')
    
    reg_IQ_b_ = cluster_fit('meanbeliefimage ~ meanbelief_priorbayesimage + mb_fracup + frac_upimage',
                            data=df_ex[(df_ex['frac_upimage'] == 0) & (df_ex['IQtask'] ==1)], group_var='ID')
    reg_IQ_g_ = cluster_fit('meanbeliefimage ~ meanbelief_priorbayesimage + mb_fracup + frac_upimage',
                            data=df_ex[(df_ex['frac_upimage'] == 1) & (df_ex['IQtask'] ==1)], group_var='ID')
    
    reg_C_b_ = cluster_fit('meanbeliefcard ~ meanbelief_priorbayescard + mb_fracupcard + frac_upcard',
                           data=df_ex[(df_ex['frac_upcard'] == 0)], group_var='ID')
    reg_C_g_ = cluster_fit('meanbeliefcard ~ meanbelief_priorbayescard + mb_fracupcard + frac_upcard',
                           data=df_ex[(df_ex['frac_upcard'] == 0)], group_var='ID')
    
    fig2_D= plt.figure(num=None, figsize=[15,15])
    fig, ax = plt.subplots()
    ax.plot(['BAD', 'GOOD'], [reg_B_b_.rsquared, reg_B_g_.rsquared], color='blue',label='Beauty')
    ax.scatter(['BAD', 'GOOD'], [reg_B_b_.rsquared, reg_B_g_.rsquared], marker =',', color='blue', s=80)

    ax.plot(['BAD', 'GOOD'], [reg_IQ_b_.rsquared, reg_IQ_g_.rsquared], color='red',label='IQ')
    ax.scatter(['BAD', 'GOOD'], [reg_IQ_b_.rsquared, reg_IQ_g_.rsquared], marker ='o', color='red', s=80)

    ax.plot(['BAD', 'GOOD'], [reg_C_b_.rsquared, reg_C_g_.rsquared], color='green',label='Control')
    ax.scatter(['BAD', 'GOOD'], [reg_C_b_.rsquared, reg_C_g_.rsquared], marker ='^', color='green', s=80)
    plt.legend()
    plt.xlabel('Condition')
    plt.xlim(-0.3,1.3)
    plt.ylabel('$R^2$')
    plt.title("Panel D. $R^2$ by condition and signal valence")
    return Extension_Figure_3
