import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


class HypothesisTest:
    def __init__(self, data):
        self.data = data

    
    def  statistical_test(self, group_a, group_b):
        '''
        Statistical test for two groups

        Parameters:
            group_a, group_b

        Returns:
            p-test, t-test: for Numerical
            chi_test: for categorical
        '''

        t_stat, p_value = ttest_ind(group_a, group_b)

        return t_stat, p_value
    

    def interpret_test(self, t_stat, p_value, hypothsis: str):
        '''
        This funcion interprets the result 
        
        Parameters:
            hypothesis(str): the hypothesis
        '''

        print(f"Province T-test: t-stat={t_stat}, p-value={p_value}")

        if p_value < 0.05:
            print(f"Reject the null hypothesis: {hypothsis}.")
        else:
            print(f"Fail to reject the null hypothesis: {hypothsis}.")