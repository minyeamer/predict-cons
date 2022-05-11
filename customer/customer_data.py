import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import date


def load_data(path='customer_data', test_size=0.3, encoding=True) -> tuple:
    """
    train, test, validation 데이터를 불러오는 함수
    """

    if path == 'original_data':
        preprocess_data()
        path = 'customer_data'

    train_data = pd.read_csv(f'{path}/train.csv')
    train_label = np.array(train_data[['target']])
    test_data = pd.read_csv(f'{path}/test.csv')
    validation_set = tuple()

    if test_size > 0:
        validation_set = make_validation_set(train_data, test_size)

    if encoding:
        train_data, test_data, validation_set = feature_encoding(train_data, test_data, validation_set)

    return train_data, test_data, train_label, validation_set


###########################################################################
############################# Preprocessing ###############################
###########################################################################


def preprocess_data():
    """
    train, test 데이터를 전처리하고 저장하는 함수
    """

    train = pd.read_csv('original_data/train.csv')
    test = pd.read_csv('original_data/test.csv')

    train, test = label_encoding(train, test)
    train, test = make_extra_variables(train, test)
    train = remove_outliers(train)
    train, test = refactor_data(train, test)

    train.to_csv('customer_data/train.csv', index=False)
    test.to_csv('customer_data/test.csv', index=False)


def label_encoding(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    문자열 범주형 데이터에 Label Encoder를 적용하는 함수
    """

    train, test = train.copy(), test.copy()

    single_list = ['Alone','YOLO','Absurd']
    train['Marital_Status'] = train['Marital_Status'].apply(lambda x: 'Single' if x in single_list else x)
    test['Marital_Status'] = test['Marital_Status'].apply(lambda x: 'Single' if x in single_list else x)

    label_features = ['Education','Marital_Status']
    for feature in label_features:
        label_encoder = LabelEncoder()
        train[feature] = label_encoder.fit_transform(train[feature])
        test[feature] = label_encoder.transform(test[feature])

    return (train, test)


def make_extra_variables(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    파생변수를 생성하고 파생변수를 제거하는 함수
    """

    train, test = train.copy(), test.copy()

    for df in [train, test]:
        df['Age'] = 2022-df['Year_Birth']
        df['Age_Range'] = (df['Age']//10).replace({2:3,8:7})

        df['Dt_Customer'] = df['Dt_Customer'].apply(lambda x: list(map(int,x.split('-'))))
        df['Dt_Customer'] = df['Dt_Customer'].apply(lambda x: date(x[2],x[1],x[0]))
        df['Days_Customer'] = df['Dt_Customer'].apply(lambda x: (date(2022,1,1)-x).days)

        max_income = df[['Age_Range','Income']].groupby('Age_Range').max()
        df['Income_Level'] = [row['Income']/max_income.loc[row['Age_Range']][0] for _,row in df.iterrows()]
        df['Income_Per'] = df['Income']/(df['Kidhome']+1)

        purchase_cat = [f'Num{t}Purchases' for t in ['Deals','Web','Catalog','Store']]
        df['NumPurchases'] = [sum(row) for _,row in df[purchase_cat].iterrows()]
        purchase_dict = {cat: i for i, cat in enumerate(purchase_cat[1:])}
        df['Perferred_Purchase'] = [purchase_dict[row.index[row.argmax()]] for _,row in df[purchase_cat[1:]].iterrows()]
        num_cat = [df[col].apply(lambda x: 7 if x > 7 else x) for col in purchase_cat+['NumWebVisitsMonth']]
        df[purchase_cat+['NumWebVisitsMonth']] = pd.DataFrame(num_cat).T

        campains = [f'AcceptedCmp{i}' for i in range(1,6)]+['Response']
        df['NumAcceptedCmp'] = sum([df[campain] for campain in campains])

    return (train, test)


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """
    이상치를 제거하는 함수
    """

    train = train.copy()

    columns = ['Year_Birth','Income']
    for column, outlier in zip(columns,[0,2]):
        cutted_data = pd.cut(train[column],bins=3,labels=[0,1,2])
        train = train[cutted_data != outlier]

    return train


def refactor_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    불필요한 열을 제거하고 전체 열을 정렬하는 함수
    """

    train, test = train.copy(), test.copy()

    [df.drop(['Year_Birth','Dt_Customer'], axis=1, inplace=True) for df in [train,test]]
    columns = train.drop(['id','target'], axis=1).columns.tolist()
    train = train.reindex(columns=['id']+sorted(columns)+['target'])
    test = test.reindex(columns=['id']+sorted(columns))

    return (train, test)


###########################################################################
############################ Train/Test Split #############################
###########################################################################


def make_validation_set(train: pd.DataFrame, test_size: int) -> tuple:
    """
    train 데이터에 Train/Test Split을 적용하는 함수
    """

    y_train = np.array(train[['target']])
    x_train = train.drop(['id','target'], axis=1)

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=0)
    
    return (x_train, x_test, y_train, y_test)


###########################################################################
############################ Feature Encoding #############################
###########################################################################


def feature_encoding(train: pd.DataFrame, test: pd.DataFrame, validation=tuple()) -> tuple:
    """
    전체 데이터에 StandardScalor와 OneHotEncoder를 적용하는 함수
    """

    if validation:
        x_train, x_test, y_train, y_test = validation

    numerical_transformer = StandardScaler()
    numerical_features = ['Age','Days_Customer','Income','Income_Level','Income_Per',
                            'NumAcceptedCmp','NumPurchases','Recency']

    categorical_transformer = OneHotEncoder(categories='auto', handle_unknown='ignore')
    categorical_features = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',
                            'Age_Range','Complain','Education','Kidhome','Marital_Status',
                            'NumCatalogPurchases','NumDealsPurchases','NumStorePurchases',
                            'NumWebPurchases','NumWebVisitsMonth','Perferred_Purchase','Response','Teenhome']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    pipe = Pipeline(steps=[('preprocessor', preprocessor)])
    train_data = pipe.fit_transform(train.drop(['id','target'], axis=1))
    test_data = pipe.transform(test)

    if validation:
        validation = (pipe.transform(x_train), pipe.transform(x_test), y_train, y_test)

    return train_data, test_data, validation
