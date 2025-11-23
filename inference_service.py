import pandas as pd
import numpy as np
import joblib


def _load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def _load_model(model_path):
    model = joblib.load(model_path)
    return model

def _load_preprocessor(preprocessor_path):
    preprocessor = joblib.load(preprocessor_path)
    return preprocessor

def _preprocess_tamping_data(data, preprocessor):
    features = [
        "beta_ballast",
        "gpr_risk",
        "geom_dev",
        "sleeper_type",
        "obstacle_flag",
        "past_defects_count",
        "noise_zone"
    ]

    data_features = data[features]
    return preprocessor.transform(data_features)

def _preprocess_risk_data(data, preprocessor):
    features = [
        "avg_beta",
        "max_geom_dev",
        "gpr_risk_max",
        "defect_density",
        "traffic_class",
        "climate_zone"
    ]

    data_features = data[features]
    return preprocessor.transform(data_features)


tamping_data_path = "./data/sleepers.csv"
risk_data_path = "./data/segments.csv"

risk_model_path = "./models/risk/model.pkl"
tamping_model_path = "./models/tamping/model.pkl"

risk_preprocessor_path = "./models/risk/preprocessor.pkl"
tamping_preprocessor_path = "./models/tamping/preprocessor.pkl"

risk_model = _load_model(risk_model_path)
tamping_model = _load_model(tamping_model_path)
risk_preprocessor = _load_preprocessor(risk_preprocessor_path)
tamping_preprocessor = _load_preprocessor(tamping_preprocessor_path)

def _join_features_and_predictions(features, predictions):
    if isinstance(predictions, pd.Series) or isinstance(predictions, pd.DataFrame):
        predictions_df = predictions
    else:
        # Es un numpy array
        predictions_df = pd.DataFrame(
            predictions, 
            index=features.index,
            columns=['prediction'] if predictions.ndim == 1 else None
        )
    return pd.concat([features, predictions_df], axis=1)


tamping_model_features = [
    "beta_ballast",
    "gpr_risk",
    "geom_dev",
    "sleeper_type",
    "obstacle_flag",
    "past_defects_count",
    "noise_zone"
]

    
risk_model_features = [
    "avg_beta",
    "max_geom_dev",
    "gpr_risk_max",
    "defect_density",
    "traffic_class",
    "climate_zone"
]

def inference_service():
    tamping_data = _load_data(tamping_data_path)
    risk_data = _load_data(risk_data_path)

    preprocessed_tamping_data = _preprocess_tamping_data(tamping_data, tamping_preprocessor)
    preprocessed_tamping_prediction = tamping_model.predict(preprocessed_tamping_data)
    joined_tamping_data = _join_features_and_predictions(tamping_data, preprocessed_tamping_prediction)

    preprocessed_risk_data = _preprocess_risk_data(risk_data, risk_preprocessor)
    preprocessed_risk_prediction = risk_model.predict(preprocessed_risk_data)
    joined_risk_data = _join_features_and_predictions(risk_data, preprocessed_risk_prediction)

    tamping_result = joined_tamping_data.to_dict(orient='records')
    risk_result = joined_risk_data.to_dict(orient='records')

    return {
        "tamping": tamping_result,
        "risk": risk_result
    }
