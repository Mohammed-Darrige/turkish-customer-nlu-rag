import sys
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No text provided"}))
        sys.exit(1)
    
    text = sys.argv[1]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    try:
        vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        duygu_model = joblib.load(os.path.join(models_dir, "duygu_svm.pkl"))
        kategori_model = joblib.load(os.path.join(models_dir, "kategori_svm.pkl"))
    except Exception as e:
        print(json.dumps({"error": f"Model loading failed: {str(e)}"}))
        sys.exit(1)
        
    try:
        # Vectorize
        X = vectorizer.transform([text])
        
        # Predict
        duygu_pred = duygu_model.predict(X)[0]
        kategori_pred = kategori_model.predict(X)[0]
        
        print(json.dumps({
            "text": text,
            "sentiment": str(duygu_pred),
            "category": str(kategori_pred)
        }))
    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
        sys.exit(1)

if __name__ == "__main__":
    main()
