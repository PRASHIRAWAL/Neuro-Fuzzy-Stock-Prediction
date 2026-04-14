import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_fuzzy_system():
    # New Antecedent/Consequent objects hold universe variables and membership
    # Universe of discourse for the score is 0 to 10
    score = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'score')
    risk = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'risk')

    # Auto-membership function population is possible with .automf(3, names=['low', 'medium', 'high'])
    # But let's define them explicitly for precise control
    score['low'] = fuzz.trapmf(score.universe, [0, 0, 3, 5])
    score['medium'] = fuzz.trimf(score.universe, [3, 5, 7])
    score['high'] = fuzz.trapmf(score.universe, [5, 7, 10, 10])

    risk['low'] = fuzz.trapmf(risk.universe, [0, 0, 3, 5])
    risk['medium'] = fuzz.trimf(risk.universe, [3, 5, 7])
    risk['high'] = fuzz.trapmf(risk.universe, [5, 7, 10, 10])

    # Fuzzy rules
    rule1 = ctrl.Rule(score['low'], risk['low'])
    rule2 = ctrl.Rule(score['medium'], risk['medium'])
    rule3 = ctrl.Rule(score['high'], risk['high'])

    # Control System
    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)
    
    return risk_sim

def evaluate_fuzzy_risk(input_score):
    """
    Evaluates the continuous risk score (0-10) using the fuzzy logic system.
    Returns the defuzzified value and the interpreted linguistic label.
    """
    risk_sim = get_fuzzy_system()
    
    # Clip input_score to valid range [0, 10]
    input_score = max(0.0, min(10.0, float(input_score)))
    
    risk_sim.input['score'] = input_score
    
    try:
        risk_sim.compute()
        output_val = risk_sim.output['risk']
    except Exception as e:
        # Fallback in case of scikit-fuzzy numerical issues
        output_val = input_score
        
    # Interpret the output
    if output_val < 3.5:
        interpreted_risk = "Low"
    elif output_val < 6.5:
        interpreted_risk = "Medium"
    else:
        interpreted_risk = "High"
        
    return output_val, interpreted_risk

if __name__ == "__main__":
    # Test cases
    for s in [2.0, 5.0, 8.5]:
        val, label = evaluate_fuzzy_risk(s)
        print(f"Input Score: {s} -> Defuzzified: {val:.2f} -> Risk: {label}")
