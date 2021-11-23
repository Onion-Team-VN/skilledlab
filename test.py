import pandas as pd 


if __name__=="__main__":
    train_df = pd.read_csv('data/qg_train.csv')
    print(train_df.head(10))