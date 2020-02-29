
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# sklearn Classes that will be changed
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def pretty_confusion(target, prediction):
    """Method for prettyfying sklearn's onboard confusion matrix."""
    
    cmc = ["Condition positive", "Condition negative"]
    cmi = ["Predicted condition positive", "Predicted condition negative"]
    
    matrix = confusion_matrix(target, prediction)
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    return pd.DataFrame(cm , columns=cmc, index=cmi)


def debug_print(X, debug):
    """Method for printing some debug information based on a debug parameter."""

    allowed = ["input", "shape", "columns", False] # False must be last
    if not debug in allowed:
        raise ValueError("Debug parameter value is not valied. Must be in ({}) or False.".format(
            ", ".join(allowed[:-1])
        ))
    elif debug=="input":
        print(X.head(5))
    elif debug=="shape":
        print(X.shape)
    elif debug=="columns":
        print(X.columns)
    elif debug is False:
        pass


# Pipeline step classes
class FeatureSelector:
    """This transformer lets you pick columns from a pandas dataset based on name"""
    
    def __init__(self, features = [], debug=False):
        self.d = debug
        self.colnames = None
        if type(features) != list:
            raise ValueError("Input features must be of type List.")
        elif type(features) == list:
            self.columns = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        debug_print(X=X, debug=self.d)
        X = X[self.columns]
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames


class NanDropper:
    """This transformer drops NaN-Rows from a pandas dataframe."""

    def __init__(self, debug=False):
        self.d = debug
        self.colnames = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        debug_print(X=X, debug=self.d)
        X = X.dropna()
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames


class NanFiller:
    """This transformer fills all missing fields with a value."""
    
    def __init__(self, filler=0, debug=False):
        self.d = debug
        self.filler = filler
        self.colnames = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        debug_print(X=X, debug=self.d)
        X = X.fillna(self.filler)
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames


class EmbarkedImputer():
    """Imputes values into Embarked variable."""

    def __init__(self, debug=False):
        self.d = debug
        self.colnames = None
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        debug_print(X=X, debug=self.d)
        df = X.copy()        
        value_to_input = df.loc[(df['fare'] < 85) & (df['fare'] > 75)  & (df['pclass'] == 1)]['embarked'].mode()
        value_to_input = value_to_input[0]
        df["embarked"] = df["embarked"].fillna(value_to_input)
        self.colnames = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.colnames


class AgeImputer():
    """Imputs age with median age of passanger's gender."""

    def __init__(self, debug=False):
        self.d = debug
        self.colnames = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        debug_print(X=X, debug=self.d)
        df = X.copy()

        df.age = df.groupby(['sex'])['age'].apply(lambda x: x.fillna(x.median()))
        self.colnames = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.colnames


class NameToTitle:
    """Replaces the name with the person's title"""

    def __init__(self, debug=False):
        self.d = debug
        self.colnames = None
    
    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        debug_print(X=X, debug=self.d)
        df = X.copy()
        title_dictionary = {
            "capt":"Officer", 
            "col":"Officer", 
            "major":"Officer", 
            "dr":"Officer",
            "jonkheer":"Royalty",
            "rev":"Officer",
            "countess":"Royalty",
            "dona":"Royalty",
            "lady":"Royalty",
            "don":"Royalty",
            "mr":"Mr",
            "mme":"Mrs",
            "ms":"Mrs",
            "mrs":"Mrs",
            "miss":"Miss",
            "mlle":"Miss",
            "master":"Royalty",
            "nan":"Mr"
        }
        splitter_function = lambda x: str.lower(x.split(", ")[1].split(".")[0])
        df["name"] = df["name"].map(splitter_function).map(title_dictionary).fillna("Mr")
        self.colnames = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.colnames


class AgeBinner:
    """Splits age into age groups and labels them""" 

    def __init__(self, debug=False):
        self.d = debug
        self.colnames = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        debug_print(X=X, debug=self.d)
        df = X.copy()
        df.age = pd.cut(
            df.age,
            bins=[0,16,30,45,150], 
            labels=["child", "young adult", "adult", "senior adult"]
        )
        self.colnames = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.colnames


class PassengerFareCalculator:
    """Calculates the fare per passanger"""

    def __init__(self, debug=False):
        self.d = debug
        self.colnames = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        debug_print(X=X, debug=self.d)
        df = X.copy()
        passenger_n = df["sibsp"] + df["parch"] + 1
        df["fare"] = df["fare"] / passenger_n
        self.colnames = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.colnames


class EmbarkMapper:
    """Inserts the real name of the embarkation station"""

    def __init__(self, debug=False):
        self.d = debug
        self.colnames = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        debug_print(X=X, debug=self.d)
        df = X.copy()
        df["embarked"] = df["embarked"].map(
            {
                "Q": "Queenstown",
                "C": "Cherbourg",
                "S": "Southhampton"
            }
        )
        self.colnames = df.columns.tolist()
        return df

    def get_feature_names(self):
        return self.colnames


class CustomOneHotEncoder:
    """OneHotEncoder that passes a labeled pandas dataframe"""

    def __init__(self, debug=False):
        self._encoder = None
        self._column_list=[]
        self.d = debug
        self.colnames = None

    def fit(self, X, y=None):
        self._encoder = OneHotEncoder(sparse=False)
        self._encoder.fit(X)
        return self

    def transform(self, X):
        self._column_list=[]
        debug_print(X=X, debug=self.d)
        result_X= self._encoder.transform(X)
        for column in self._encoder.categories_:
            for value in column:
                self._column_list.append(value)
        X = pd.DataFrame(result_X, columns=self._column_list)
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames


class CustomSimpleImputer:
    """SimpleScaler that passes a labeled pandas dataframe"""

    def __init__(self, debug=False, strategy="median"):
        self._imputer = None
        self._column_list=[]
        self.d = debug
        self._strategy = strategy
        self.colnames = None

    def fit(self, X, y=None):
        self._imputer = SimpleImputer(strategy=self._strategy)
        self._imputer.fit(X)
        return self

    def transform(self, X):
        self._column_list=[]
        debug_print(X=X, debug=self.d)
        result_X= self._imputer.transform(X)
        for column in X.columns:
            self._column_list.append(column)
        pd.DataFrame(result_X, columns=self._column_list)
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames


class CustomRobustScaler:
    """RobustScaler that passes a labeled pandas dataframe"""

    def __init__(self, debug=False, strategy="median"):
        self._scaler = None
        self._column_list=[]
        self.d = debug
        self.colnames = None

    def fit(self, X, y=None):
        self._scaler = RobustScaler()
        self._scaler.fit(X)
        return self

    def transform(self, X):
        self._column_list=[]
        debug_print(X=X, debug=self.d)
        result_X= self._scaler.transform(X)
        for column in X.columns:
            self._column_list.append(column)
        X = pd.DataFrame(result_X, columns=self._column_list)
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames



class TravelAloneMapper:
    """Pipeline step adds a new travel-alone attribute"""

    def __init__(self):
        self.colnames = None
    
    def fit(self, X, y=None):
        if "sibsp" not in X.columns:
            raise ValueError("X does not contain sibsp")
        if "parch" not in X.columns:
            raise ValueError("X does not contain parch")
        return self
    
    def transform(self, X):
        X["travel_alone"] = None
        X["travel_alone"] = X.loc[
            (X["sibsp"] == 0) & (X["parch"] == 0)
        ]["travel_alone"].map(lambda x: 1)
        X["travel_alone"] = X["travel_alone"].fillna(0)
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames


class KidWithTwoParentsMapper:
    """Pipeline step adds a new kid_with_two_parents attribute"""

    def __init__(self, age=12):
        self.age_th = age
        self.colnames = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X["kid_with_two_parents"] = (X.age < self.age_th) & (X.parch == 2)
        X["kid_with_two_parents"] = X["kid_with_two_parents"].map({True: 1, False: 0})
        self.colnames = X.columns.tolist()
        return X
    
    def get_feature_names(self):
        return self.colnames


class FeaturePipeline(Pipeline):
    """Subclass of sklearn.Pipeline having a get_feature_names method."""

    def get_feature_names(self):
        return self.steps[-1][1].get_feature_names()
