import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(plot:bool):
    data = pd.read_csv("./data/HAM10000_metadata.csv")
    
    print(f'\n{data.shape[0]} samples before dropping patients NaN values')
    data = data.dropna()
    data = data.reset_index(drop=True)
    print(f'{data.shape[0]} samples after dropping patients NaN values\n')
    
    if plot:
        pd.set_option('display.max_colwidth', None)
        print(data.head())
        
        fig, axes = plt.subplots(1,3, figsize=(20,5))

        sns.histplot(data=data, bins= 10, x = 'age', kde=True, color=sns.color_palette("crest")[3], ax=axes[0])
        axes[0].set_title('Age')

        sns.countplot(data=data, x = 'sex', hue='sex', palette='deep', ax=axes[1])
        axes[1].set_title("Sex")

        sns.countplot(data=data, x = 'dx', palette='deep', hue = 'dx', ax=axes[2])
        axes[2].set_title('Diagnoses')

        plt.suptitle('HAM10000 Data Distribution', fontweight='bold', fontsize=14)
        plt.show()
        
        plt.figure(figsize=(20,9))
        sns.countplot(data=data, x='localization', palette='deep', hue='localization')
        plt.title('Localization Area', fontweight='bold',fontsize=14)
        plt.show()
    
    return data