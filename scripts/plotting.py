import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotting:
    def __init__(self, data):
        self.data = data

    def plot_cover_type_by_province(self):
        '''
        This funcion plots the cover type distrubution by Province
        '''
        cover_counts = self.data.groupby('Province')['CoverType'].value_counts().unstack()

        
        cover_counts['Total'] = cover_counts.sum(axis=1)
        cover_counts_sorted = cover_counts.sort_values(by='Total', ascending=False).drop(columns='Total')

        cover_counts_sorted.plot(
            kind='bar',
            stacked=True,
            figsize=(12, 8),
            color=sns.color_palette("pastel")  
        )

        
        plt.title('Cover Type Distribution by Province', fontsize=16)
        plt.xlabel('Province', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Cover Type')
        plt.tight_layout()  
        
        plt.show()

    def distribution_by_vehicle(self, col:str):
        '''
        Plots Distribution of a column by Vehicle Type
        '''

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='VehicleType', y=col, data=self.data)
        plt.title(f'Distribution of {col} by Vehicle Type', fontsize=16)
        plt.xlabel('Vehicle Type', fontsize=14)
        plt.ylabel('Total Premium', fontsize=14)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()


    def gender_distibution(self):
        '''
        Plots the gender distribution of clients
        '''
        gender_counts = self.data['Gender'].value_counts()

        
        plt.figure(figsize=(6, 4))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Gender Distribution of Clients', fontsize=16)
        plt.axis('equal')  
        plt.tight_layout()
        plt.show()