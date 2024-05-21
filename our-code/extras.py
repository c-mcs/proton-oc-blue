import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def read_csv_data( filename: str) -> pd.DataFrame:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(current_directory, filename + ".csv"))

def df_to_dict(df: pd.DataFrame, extra_depth: bool = False):
    """
    Based on the number of pandas DataFrame columns, transforms the dataframe into nested dictionaries as follows:
    df-columns = age, sex, education, p --> dict-keys = {age:{sex:[education, p]}}

    If extra_depth is True the transformation has an extra level of depth as follows:
    df-columns = age, sex, education, p --> dict-keys = {age:{sex:{education: p}}}

    This transformation ensures a faster access to the values using the dictionary keys.
    :param df: pd.DataFrame, the df to be transformed
    :param extra_depth: bool, if True gives an extra level of depth
    :return: Dict, a new dictionary
    """
    dic = dict()
    extra_depth_modifier = 0
    if extra_depth:
        extra_depth_modifier = 1
    if len(df.columns) + extra_depth_modifier == 2:
        for col in np.unique(df.iloc[:, 0]):
            dic[col] = df[df.iloc[:, 0] == col].iloc[:, 1].values
    if len(df.columns) + extra_depth_modifier == 3:
        for col in np.unique(df.iloc[:, 0]):
            dic[col] = df[df.iloc[:, 0] == col].iloc[:, 1:].values
    if len(df.columns) + extra_depth_modifier == 4:
        for col in np.unique(df.iloc[:, 0]):
            dic[col] = df[df.iloc[:, 0] == col].iloc[:, 1:]
        for key in dic:
            subdic = dict()
            for subcol in np.unique(dic[key].iloc[:, 0]):
                if extra_depth:
                    subdic[subcol] = dic[key][dic[key].iloc[:, 0] == subcol].iloc[:, 1:].values[0][0]
                else:
                    subdic[subcol] = dic[key][dic[key].iloc[:, 0] == subcol].iloc[:, 1:].values
            dic[key] = subdic
    return dic

def weighted_n_of(n, agentset,
                  weight_function, rng_istance):
    """
    Given a set or List of agents @agentset an integer @n and a lambda function @weight_function.
    This function performs a weighted extraction, without replacing based on the lambda function.
    This procedure takes into account negative numbers and weights equal to zero.
    :param n: int
    :param agentset: Union[List[Person], Set[Person]]
    :param weight_function: Callable
    :param rng_istance: numpy.random.default_rng
    :return: List[Person]
    """
    p = [float(weight_function(x)) for x in agentset]
    for pi in p:
        if pi < 0:
            min_value = np.min(p)
            p = [i - min_value for i in p]
            break
    sump = sum(p)
    #if there are more zeros than n required in p
    if np.count_nonzero(p) < n:
        n = np.count_nonzero(p)
    #If there are only zeros
    if sump == 0:
        p = None
    else:
        p = [i/sump for i in p]
    #If the type is wrong
    if type(agentset) != list:
        agentset = list(agentset)
    return rng_istance.choice(agentset, int(n), replace=False, p=p)

def weighted_one_of(agentset,
                    weight_function, rng_istance):
    return weighted_n_of(1, agentset, weight_function, rng_istance)[0]

def pick_from_pair_list(a_list_of_pairs,
                        rng_istance):
    """
    given a list of pairs, containing an object and a probability (e.g. [[object, p],[object, p]])
    return an object based on the probability(p)
    :param a_list_of_pairs:list, a list of pairs (e.g. [[object, p],[object, p]])
    :param rng_istance: numpy.random instance,
    :return: object
    """
    return weighted_one_of(a_list_of_pairs, lambda x: x[-1], rng_istance)[0]

def fit_polynomial_regression(df, target_column, degree=2):
    """
    Fit a polynomial regression model to the data in the DataFrame, save the equation, and provide prediction capability.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_column (str): The name of the target column in the DataFrame.
    degree (int): The degree of the polynomial features. Default is 2.
    
    Returns:
    model (LinearRegression): The fitted polynomial regression model.
    poly_features (PolynomialFeatures): The polynomial features used for the transformation.
    equation (str): The polynomial regression equation.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predictions
    y_pred = model.predict(X_poly)
    
    # Evaluation metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Extracting the equation
    intercept = model.intercept_
    coefs = model.coef_
    feature_names = poly.get_feature_names_out(X.columns)
    
    equation = f"{intercept:.4f}"
    for coef, feature in zip(coefs, feature_names):
        equation += f" + ({coef:.4f} * {feature})"
    
    print("Equation:", equation)
    
    return model, poly, equation

def predict_with_equation(model, poly, new_data):
    """
    Predict using the fitted polynomial regression model and the polynomial features.
    
    Parameters:
    model (LinearRegression): The fitted polynomial regression model.
    poly_features (PolynomialFeatures): The polynomial features used for the transformation.
    new_data (pd.DataFrame): The new data to predict on.
    
    Returns:
    np.ndarray: The predicted values.
    """
    # Transform new data with polynomial features
    new_data_poly = poly.transform(new_data)
    
    # Predict
    predictions = model.predict(new_data_poly)
    
    return predictions

def pick_from_population_pool_by_age_and_gender(age_wanted,
                                                male_wanted,
                                                population, random):
    """
    Pick an agent with specific age and sex, None otherwise
    :param age_wanted: int, age wanted
    :param male_wanted: bool, gender wanted
    :param population: List[Person], the population
    :return: Union[Person, None], the agent or None
    """
    if not [x for x in population if x.gender_is_male == male_wanted and x.age == age_wanted]:
        return None
    picked_person = random.choice(
        [x for x in population if x.gender_is_male == male_wanted and x.age == age_wanted])
    population.remove(picked_person)
    return picked_person

def pick_from_population_pool_by_age(age_wanted,
                                     population,random):
    """
    Pick an agent with specific age form population, None otherwise
    :param age_wanted: int, age wanted
    :param population: List[Person], the population
    :return: agent or None
    """
    if age_wanted not in [x.age for x in population]:
        return None
    picked_person = random.choice([x for x in population if x.age == age_wanted])
    population.remove(picked_person)
    return picked_person

def df_to_lists(df: pd.DataFrame, split_row: bool = True):
    """
    This function transforms a pandas DataFrame into nested lists as follows:
    df-columns = age, sex, education, p --> list = [[age,sex],[education,p]]

    This transformation ensures a faster access to the values using the position in the list
    :param df: pandas df, the df to be transformed
    :param split_row: bool, default = True
    :return: list, a new list
    """
    output_list = list()
    if split_row:
        temp_list = df.iloc[:, :2].values.tolist()
        for index, row in df.iterrows():
            output_list.append([temp_list[index], [row.iloc[2], row.iloc[3]]])
    else:
        output_list = df.values.tolist()
    return output_list

def list_contains_problems(ego, candidates):
    """
    This procedure checks if there are any links between partners within the candidate pool.
    Returns True if there are, None if there are not. It is used during ProtonOc.setup_siblings
    procedure to avoid incestuous marriages.
    :param ego: Person, the agent
    :param candidates: Union[List[Person], Set[Person]], the candidates
    :return: Union[bool, None], True if there are links between partners, None otherwise.
    """
    all_potential_siblings = [ego] + ego.get_neighbor_list("sibling") + candidates + [sibling for candidate in
                                                                                      candidates for sibling in
                                                                                      candidate.neighbors.get(
                                                                                          'sibling')]
    for sibling in all_potential_siblings:
        if sibling.get_neighbor_list("partner") and sibling.get_neighbor_list("partner")[
            0] in all_potential_siblings:
            return True