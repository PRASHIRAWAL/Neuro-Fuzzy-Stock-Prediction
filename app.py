from flask import Flask, render_template, request, jsonify
import logging
from neuro_fuzzy import predict_neuro_fuzzy_risk

# Setup robust logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyzer')
def analyzer():
    return render_template('analyzer.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Parse Input
        data = request.get_json()
        if not data:
            logger.warning("Empty request data received.")
            return jsonify({'error': 'No input data provided'}), 400
            
        # 2. Input Validation (Missing fields)
        required_fields = ['volatility', 'trend', 'volume']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return jsonify({'error': f"Missing required field: {field}"}), 400
                
        # 3. Type Validation & Conversion
        try:
            volatility = float(data['volatility'])
            trend = float(data['trend'])
            volume = float(data['volume'])
        except (ValueError, TypeError):
            logger.warning("Invalid data types for input fields.")
            return jsonify({'error': 'All fields must be valid numeric values'}), 400
            
        # 4. Predict
        logger.info(f"Predicting risk for Vola:{volatility}, Trend:{trend}, Vol:{volume}")
        result = predict_neuro_fuzzy_risk(volatility, trend, volume)
        
        logger.info(f"Prediction successful: {result['predicted_risk']}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Internal Server Error: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred while processing the prediction.'}), 500

if __name__ == '__main__':
    # Using threaded=True for better handling of requests out of the box
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
