"""
Experiment Configurations for Research_test Folder
===================================================

Matches your folder structure:
D:/IIT/Year 4/FYP/Datasets/GIFGIF_lucas/Research_test/
"""

from pathlib import Path

# ============================================================================
# BASE PATHS - AUTOMATICALLY SET
# ============================================================================

BASE_DIR = Path(r"D:/IIT/Year 4/FYP/Datasets/GIFGIF_lucas")
RESEARCH_DIR = BASE_DIR / "Research_test"
CSV_DIR = RESEARCH_DIR / "csvs"
GIF_DIR = BASE_DIR / "Data/gifgif-images-v1/gifgif-images"

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_experiment_config(experiment_name, data_dir=None):
    """
    Get configuration for specific experiment
    
    Args:
        experiment_name: One of exp1_all, exp2_13, exp3_6groups, exp4_3groups
        data_dir: Optional override for CSV directory (default: Research_test/csvs)
    """
    
    if data_dir is None:
        data_dir = CSV_DIR
    else:
        data_dir = Path(data_dir)
    
    configs = {
        # ====================================================================
        # EXPERIMENT 1: All Emotions from Original Dataset
        # ====================================================================
        'exp1_all_emotions': {
            'num_classes': None,  # Will be determined from CSV
            'emotion_to_idx': None,  # Will be determined from CSV
            'train_csv': data_dir / 'train_all_emotions.csv',
            'val_csv': data_dir / 'val_all_emotions.csv',
            'test_csv': data_dir / 'test_all_emotions.csv',
            'description': 'All emotions from original GIFGIF dataset'
        },
        
        # ====================================================================
        # EXPERIMENT 2: 13 Filtered Emotions (>= 150 samples)
        # ====================================================================
        'exp2_13_emotions': {
            'num_classes': 13,
            'emotion_to_idx': {
                'happiness': 0,
                'excitement': 1,
                'amusement': 2,
                'pride': 3,  # Similar to awe
                'contentment': 4,
                'satisfaction': 5,
                'anger': 6,
                'fear': 7,
                'disgust': 8,
                'sadness': 9,
                'embarrassment': 10,
                'surprise': 11,
                'contempt': 12
            },
            'train_csv': data_dir / 'train_13_emotions.csv',
            'val_csv': data_dir / 'val_13_emotions.csv',
            'test_csv': data_dir / 'test_13_emotions.csv',
            'description': '13 emotions with sufficient samples (>= 150)'
        },
        
        # ====================================================================
        # EXPERIMENT 3: 6 Hierarchical Groups (YOUR MAIN CONTRIBUTION) ⭐
        # ====================================================================
        'exp3_6_groups': {
            'num_classes': 6,
            'emotion_to_idx': {
                'contempt': 0,
                'negative_intense': 1,
                'negative_subdued': 2,
                'positive_calm': 3,
                'positive_energetic': 4,
                'surprise': 5
            },
            'train_csv': data_dir / 'train_6_groups.csv',
            'val_csv': data_dir / 'val_6_groups.csv',
            'test_csv': data_dir / 'test_6_groups.csv',
            'description': '6 psychologically-grounded hierarchical groups'
        },
        
        # ====================================================================
        # EXPERIMENT 4: 3 Simple Groups
        # ====================================================================
        'exp4_3_groups': {
            'num_classes': 3,
            'emotion_to_idx': {
                'positive': 0,
                'negative': 1,
                'neutral': 2
            },
            'train_csv': data_dir / 'train_3_groups.csv',
            'val_csv': data_dir / 'val_3_groups.csv',
            'test_csv': data_dir / 'test_3_groups.csv',
            'description': '3 simple groups (positive/negative/neutral)'
        }
    }
    
    # Handle alternative naming
    if experiment_name == 'exp3_6groups':
        experiment_name = 'exp3_6_groups'
    elif experiment_name == 'exp4_3groups':
        experiment_name = 'exp4_3_groups'
    
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(configs.keys())}")
    
    config = configs[experiment_name]
    
    # Auto-detect emotion_to_idx if not specified (for Exp 1)
    if config['emotion_to_idx'] is None and config['train_csv'].exists():
        import pandas as pd
        df = pd.read_csv(config['train_csv'])
        unique_emotions = sorted(df['emotion_label'].unique())
        config['emotion_to_idx'] = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
        config['num_classes'] = len(unique_emotions)
    
    return config

# ============================================================================
# EMOTION GROUPING MAPPINGS (For Reference)
# ============================================================================

# Map original emotions → 6 hierarchical groups
EMOTION_TO_6GROUPS = {
    # Positive energetic (high arousal, positive valence)
    'happiness': 'positive_energetic',
    'excitement': 'positive_energetic',
    'amusement': 'positive_energetic',
    'pride': 'positive_energetic',
    
    # Positive calm (low arousal, positive valence)
    'contentment': 'positive_calm',
    'satisfaction': 'positive_calm',
    'relief': 'positive_calm',
    
    # Negative intense (high arousal, negative valence)
    'anger': 'negative_intense',
    'fear': 'negative_intense',
    'disgust': 'negative_intense',
    
    # Negative subdued (low arousal, negative valence)
    'sadness': 'negative_subdued',
    'embarrassment': 'negative_subdued',
    'shame': 'negative_subdued',
    'guilt': 'negative_subdued',
    
    # Surprise (high arousal, neutral/mixed valence)
    'surprise': 'surprise',
    
    # Contempt (unique category)
    'contempt': 'contempt'
}

# Map 6 groups → 3 simple groups
GROUPS6_TO_3GROUPS = {
    'positive_energetic': 'positive',
    'positive_calm': 'positive',
    'negative_intense': 'negative',
    'negative_subdued': 'negative',
    'surprise': 'neutral',
    'contempt': 'neutral'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_gif_dir():
    """Get GIF directory path"""
    return GIF_DIR

def get_csv_dir():
    """Get CSV directory path"""
    return CSV_DIR

def get_results_dir():
    """Get results directory path"""
    results_dir = RESEARCH_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def get_models_dir():
    """Get models directory path"""
    models_dir = RESEARCH_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def list_available_experiments():
    """List all available experiments"""
    experiments = [
        'exp1_all_emotions',
        'exp2_13_emotions',
        'exp3_6_groups',  # MAIN CONTRIBUTION
        'exp4_3_groups'
    ]
    return experiments

def print_experiment_info(experiment_name):
    """Print information about an experiment"""
    try:
        config = get_experiment_config(experiment_name)
        
        print(f"\n{'='*70}")
        print(f"📊 EXPERIMENT: {experiment_name}")
        print(f"{'='*70}")
        print(f"Description: {config['description']}")
        print(f"Number of classes: {config['num_classes']}")
        print(f"Emotions: {list(config['emotion_to_idx'].keys())}")
        print(f"\nCSV Files:")
        print(f"  Train: {config['train_csv']}")
        print(f"  Val:   {config['val_csv']}")
        print(f"  Test:  {config['test_csv']}")
        
        # Check if files exist
        all_exist = all([
            config['train_csv'].exists(),
            config['val_csv'].exists(),
            config['test_csv'].exists()
        ])
        
        if all_exist:
            print(f"\n✅ All CSV files exist and ready to use!")
        else:
            print(f"\n⚠️  Some CSV files are missing. Run setup_research_test.py first!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🔧 EXPERIMENT CONFIGURATIONS")
    print("="*70)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Research Directory: {RESEARCH_DIR}")
    print(f"CSV Directory: {CSV_DIR}")
    print(f"GIF Directory: {GIF_DIR}")
    
    print(f"\n{'='*70}")
    print("📋 AVAILABLE EXPERIMENTS")
    print(f"{'='*70}")
    
    for exp_name in list_available_experiments():
        print_experiment_info(exp_name)
    
    print(f"\n{'='*70}")
    print("✅ CONFIGURATION MODULE READY")
    print(f"{'='*70}")
