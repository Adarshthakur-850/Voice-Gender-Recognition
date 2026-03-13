from src.data_loader import generate_synthetic_data
from src.train import train_pipeline
from src.visualization import evaluate_and_plot, plot_history
import pickle
import os

def main():
    print("Starting Voice Gender Recognition Pipeline...")
    
    # 1. Data Generation
    generate_synthetic_data()
    
    # 2. Train and Evaluate
    # Running small epochs for demonstration
    model, X_test, y_test = train_pipeline(epochs=5)
    
    if model:
        # 3. Features & Viz
        evaluate_and_plot(model, X_test, y_test)
        
        # Plot History
        if os.path.exists('models/history.pkl'):
            with open('models/history.pkl', 'rb') as f:
                history = pickle.load(f)
            plot_history(history)
            
        print("Pipeline completed. Use 'python src/inference.py' for real-time test.")

if __name__ == "__main__":
    main()
