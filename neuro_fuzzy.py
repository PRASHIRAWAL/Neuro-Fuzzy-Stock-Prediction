import joblib
import os
import numpy as np
from fuzzy_logic import evaluate_fuzzy_risk

# Global cache for the machine learning model and scaler
_model = None
_scaler = None

def load_models():
    """Load the neural network and the scaler from disk."""
    global _model, _scaler
    if _model is None or _scaler is None:
        model_dir = 'model'
        _model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        _scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

def predict_neuro_fuzzy_risk(volatility, trend, volume):
    """
    Main integration function.
    1. Scales the features
    2. Gets probabilities from the Neural Network
    3. Transforms probabilities to a 0-10 score
    4. Evaluates through the Fuzzy System
    5. Calculates mock feature contributions for explainability
    """
    load_models()
    
    # 1. Feature Scaling
    features = np.array([[volatility, trend, volume]])
    scaled_features = _scaler.transform(features)
    
    # 2. Neural Network Prediction (Probability-based)
    # MLPClassifier provides predict_proba predicting the classes we mapped (0: Low, 1: Medium, 2: High)
    probs = _model.predict_proba(scaled_features)[0]
    
    prob_low = probs[0] if len(probs) > 0 else 0
    prob_med = probs[1] if len(probs) > 1 else 0
    prob_high = probs[2] if len(probs) > 2 else 0
    
    # 3. Convert Probability distribution into a continuous risk score (0-10 scale)
    # Low = ~1.5 impact, Medium = ~5.0 impact, High = ~8.5 impact
    continuous_score = (prob_low * 1.5) + (prob_med * 5.0) + (prob_high * 8.5)
    
    # Ensure it's bounded safely
    continuous_score = max(0.0, min(10.0, continuous_score))
    
    # --- Explainability ---
    # We use a rough heuristic using the first layer weights to find what influenced the network
    weights = np.abs(_model.coefs_[0].sum(axis=1)) # approximate magnitude of influence
    # Multiply by absolute scaled features
    impacts = np.abs(scaled_features[0]) * weights
    total_impact = np.sum(impacts) + 1e-9
    
    vola_cont = (impacts[0] / total_impact) * 100
    trend_cont = (impacts[1] / total_impact) * 100
    vol_cont = (impacts[2] / total_impact) * 100
    
    conts = {'Volatility': vola_cont, 'Trend': trend_cont, 'Volume': vol_cont}
    top_feature = max(conts, key=conts.get)
    # ----------------------
    
    # 4. Fuzzy logic evaluation
    defuzz_val, final_risk = evaluate_fuzzy_risk(continuous_score)
    
    reasons = []
    
    # 1. Feature Intuition
    reasons.append(f"<b>{top_feature}</b> was the dominant feature driving {conts[top_feature]:.1f}% of the model's analytical weight.")
    
    if top_feature == 'Volatility' and volatility > 100:
        reasons.append("High price instability caused the model to flag elevated risk.")
    elif top_feature == 'Trend' and trend < -10:
        reasons.append("The bearish trend weighed heavily on the risk outlook.")
    elif top_feature == 'Trend' and trend > 10:
        reasons.append("The bullish upward trend provided strong downside protection.")
        
    # 2. Neural Net Intuition
    nn_confidence = max(prob_low, prob_med, prob_high) * 100
    reasons.append(f"The Neural Network classified this pattern natively as <i>{final_risk} Risk</i> with <b>{nn_confidence:.1f}%</b> confidence.")
    
    # 3. Fuzzy Logic Intuition
    reasons.append(f"Finally, the Fuzzy Inference Engine evaluated those probabilities and locked the output to a rigid <b>{defuzz_val:.1f}/10</b> score.")
    
    explanation_msg = "<br><br>• ".join(["<b>How the AI arrived at this:</b>"] + reasons)
    
    result = {
        'continuous_score': round(continuous_score, 2),
        'defuzzified_value': round(defuzz_val, 2),
        'predicted_risk': final_risk,
        'probabilities': {
            'Low': round(prob_low, 3),
            'Medium': round(prob_med, 3),
            'High': round(prob_high, 3)
        },
        'explainability': {
            'message': explanation_msg,
            'contributions': {
                'Volatility': round(vola_cont, 1),
                'Trend': round(trend_cont, 1),
                'Volume': round(vol_cont, 1)
            }
        }
    }
    
    return result

if __name__ == "__main__":
    # Test stub (will fail if model isn't trained yet)
    try:
        res = predict_neuro_fuzzy_risk(volatility=15.0, trend=-5.0, volume=0.8)
        print("Neuro-Fuzzy Output:", res)
    except Exception as e:
        print("Error during test run (model might not be trained):", e)
