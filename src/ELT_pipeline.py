import luigi
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import logging
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Task 1: Extract Data from Original Datafile
class ExtractDataFromFile(luigi.Task):
    def output(self):
        return luigi.LocalTarget('Data/Original/adult-depression-lghc-indicator-24.csv')
    
    def run(self):
        if not self.output().exists():
            raise FileNotFoundError(f"Expected file not found: {self.output().path}")

# Helper functions for data transformation and pickle operations
def transform_data(df, strata_type):
    return df[df['Strata'] == strata_type].groupby(['Year', 'Strata Name'], as_index=False)['Percent'].mean()

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def process_and_save(df, strata, strata_type, output_path):
    transformed_data = transform_data(df, strata_type)
    save_pickle(transformed_data, output_path)

# Task 2: Transform Data by Strata Type using pickle files
class TransformDataByStrata(luigi.Task):
    def requires(self):
        return ExtractDataFromFile()
    
    def output(self):
        strata_types = ['total', 'sex', 'race', 'education', 'income', 'age']
        return {strata: luigi.LocalTarget(f'Data/Temp/{strata}_data.pkl') for strata in strata_types}

    def run(self):
        df = pd.read_csv(self.input().path)

        required_columns = {'Year', 'Strata', 'Strata Name'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Input data missing columns: {required_columns - set(df.columns)}")

        os.makedirs('Data/Temp', exist_ok=True)

        # Map strata to their corresponding transformation types
        strata_mapping = {
            'total': 'Total',
            'sex': 'Sex',
            'race': 'Race-Ethnicity',
            'education': 'Education',
            'income': 'Income',
            'age': 'Age'
        }

        # Execute transformations in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_and_save, df, strata, strata_type, self.output()[strata].path)
                for strata, strata_type in strata_mapping.items()
            ]
            for future in futures:
                future.result() 

# Task 3: Load Transformed Data into CSV Files
class LoadTransformedData(luigi.Task):
    def requires(self):
        return TransformDataByStrata()

    def output(self):
        output_paths = {
            'total': 'Data/Transformed/total_data.csv',
            'sex': 'Data/Transformed/sex_data.csv',
            'race': 'Data/Transformed/race_ethnicity_data.csv',
            'education': 'Data/Transformed/education_data.csv',
            'income': 'Data/Transformed/income_data.csv',
            'age': 'Data/Transformed/age_data.csv'
        }
        # return a luigi.LocalTarget for all those paths.
        return {strata: luigi.LocalTarget(path) for strata, path in output_paths.items()}

    def run(self):
        os.makedirs('Data/Transformed', exist_ok=True)
        for strata, target in self.output().items():
            data = load_pickle(self.input()[strata].path)
            data.to_csv(target.path, index=False)
            os.remove(self.input()[strata].path)

# Function to create and save plot
def create_and_save_plot(data:pd.DataFrame, title, xlabel, ylabel, file_path):
    # Set plot size
    plt.figure(figsize=(10, 6))

    # creates group names to display for lines for each unique name
    for group in data['Strata Name'].unique():
        # Creates a subset dataframe with rows that match groups 
        subset = data[data['Strata Name'] == group]

        # Adds subeset to plot. Repeating just means more data is added with this.
        plt.plot(subset['Year'], subset['Percent'], marker='o', label=group)
    
    # sets rest of the important data
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(title='Group')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

# Optional Task: Make plots for Loaded Data
class VisualizeData(luigi.Task):
    def requires(self):
        return LoadTransformedData()

    def output(self):
        plot_paths = {
            'total': 'Output/plots/total_plot.png',
            'sex': 'Output/plots/sex_plot.png',
            'race': 'Output/plots/race_ethnicity_plot.png',
            'education': 'Output/plots/education_plot.png',
            'income': 'Output/plots/income_plot.png',
            'age': 'Output/plots/age_plot.png'
        }
        return {strata: luigi.LocalTarget(path) for strata, path in plot_paths.items()}

    def run(self):
        os.makedirs('Output/plots', exist_ok=True)
        with ProcessPoolExecutor() as executor:
            futures = []
            for strata, target in self.output().items():
                csv_path = self.input()[strata].path
                data = pd.read_csv(csv_path)
                title = f"Average Percent of Depressive Adults by {strata.capitalize()} Over Years"
                futures.append(executor.submit(create_and_save_plot, data, title, 'Year', 'Average Percent', target.path))
            for future in futures:
                future.result()  # Ensure completion and catch potential exceptions

# Starting the pipeline
if __name__ == "__main__":
    luigi.build([LoadTransformedData(), VisualizeData()], local_scheduler=True)
