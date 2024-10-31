import luigi
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor

# Task 1: Extract Data from Original Datafile
class ExtractDataFromFile(luigi.Task):
    def output(self):
        return luigi.LocalTarget('Data/Original/adult-depression-lghc-indicator-24.csv')
    
    def run(self):
        if not self.output().exists():
            raise FileNotFoundError(f"Expected file not found: {self.output().path}")

# Function to transform data based on Strata type from dataframe
def transform_data(df, strata_type):
    return df[df['Strata'] == strata_type].groupby(['Year', 'Strata Name'], as_index=False)['Percent'].mean()

# Task 2: Transform Data by Strata Type (in-memory only, no saving)
class TransformDataByStrata(luigi.Task):
    def requires(self):
        return ExtractDataFromFile()

    def output(self):
        # Outputs a marker to indicate that the transformation is complete
        return luigi.LocalTarget('Data/Transformed/transformation_complete.marker')

    def run(self):
        df = pd.read_csv(self.input().path)

        if 'Year' not in df.columns or 'Strata' not in df.columns or 'Strata Name' not in df.columns:
            raise ValueError("Input data does not contain required columns.")

        # Perform the transformation in-memory
        self.age_data = transform_data(df, 'Age')
        self.race_data = transform_data(df, 'Race-Ethnicity')

        # Save a marker to show that transformation is complete
        os.makedirs('Data/Transformed', exist_ok=True)
        with open(self.output().path, 'w') as f:
            f.write("Transformation complete.\n")

# Task 3: Load Transformed Data into Files
class LoadTransformedData(luigi.Task):
    def requires(self):
        return TransformDataByStrata()

    def output(self):
        return {
            'transformed_age_data': luigi.LocalTarget('Data/Transformed/age_group_data.csv'),
            'transformed_race_data': luigi.LocalTarget('Data/Transformed/race_ethnicity_data.csv')
        }

    def run(self):
        # Access in-memory transformed data from the TransformDataByStrata task
        transform_task = self.requires()
        age_data = transform_task.age_data
        race_data = transform_task.race_data

        # Save transformed data to CSV files in Data/Transformed
        os.makedirs('Data/Transformed', exist_ok=True)
        age_data.to_csv(self.output()['transformed_age_data'].path, index=False)
        race_data.to_csv(self.output()['transformed_race_data'].path, index=False)

def create_and_save_plot(data, title, xlabel, ylabel, file_path):
    plt.figure(figsize=(10, 6))
    for group in data['Strata Name'].unique():
        subset = data[data['Strata Name'] == group]
        plt.plot(subset['Year'], subset['Percent'], marker='o', label=group)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(title='Group')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


# Optional Task: Visualize Loaded Data
class VisualizeData(luigi.Task):
    def requires(self):
        return LoadTransformedData()

    def output(self):
        return {
            'age_group_plot': luigi.LocalTarget('Output/plots/age_group_plot.png'),
            'race_ethnicity_plot': luigi.LocalTarget('Output/plots/race_ethnicity_plot.png')
        }

    def run(self):
        os.makedirs('Output/plots', exist_ok=True)

        # Load data from the final loaded files
        age_data = pd.read_csv(self.input()['transformed_age_data'].path)
        race_data = pd.read_csv(self.input()['transformed_race_data'].path)

        # Generate and save plots concurrently
        with ProcessPoolExecutor() as executor:
            executor.submit(create_and_save_plot, age_data,
                            'Average Percent of Depressive Adults by Age Group Over Years',
                            'Year', 'Average Percent', self.output()['age_group_plot'].path)

            executor.submit(create_and_save_plot, race_data,
                            'Average Percent of Depressive Adults by Race-Ethnicity Over Years',
                            'Year', 'Average Percent', self.output()['race_ethnicity_plot'].path)

# Starting the pipeline
if __name__ == "__main__":
    luigi.build([LoadTransformedData(), VisualizeData()], local_scheduler=True) # 
