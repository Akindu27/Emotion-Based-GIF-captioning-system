"""
Complete Setup Script for Research_test Folder
===============================================

This script will:
1. Create all necessary directories
2. Process your original CSV to create train/val/test splits
3. Generate CSVs for all 4 experiments
4. Verify everything is ready

Run this ONCE to set up everything!

Usage:
    python setup_research_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS!
# ============================================================================

BASE_DIR = Path(r"D:/IIT/Year 4/FYP/Datasets/GIFGIF_lucas")
RESEARCH_DIR = BASE_DIR / "Research_test"
ORIGINAL_CSV = BASE_DIR / "gifgif_emotion_labels.csv"
GIF_DIR = BASE_DIR / "Data/gifgif-images-v1/gifgif-images"

# ============================================================================
# EMOTION MAPPINGS
# ============================================================================

# All 20 original emotions in your CSV
ALL_EMOTIONS = [
    'pleasure', 'disgust', 'happiness', 'pride', 'excitement',
    'embarrassment', 'surprise', 'sadness', 'fear', 'satisfaction',
    'guilt', 'contempt', 'shame', 'anger', 'amusement',
    'contentment', 'relief', 'craving', 'horror', 'sexual_desire'
]

# Emotions with sufficient samples (>= 150)
# Based on your previous work, these are the 13 emotions you kept
FILTERED_EMOTIONS = [
    'happiness', 'excitement', 'amusement', 'awe',
    'contentment', 'satisfaction',
    'anger', 'fear', 'disgust',
    'sadness', 'embarrassment',
    'surprise', 'contempt'
]

# Map to 6 hierarchical groups (your main contribution)
EMOTION_TO_6GROUPS = {
    # Positive energetic
    'happiness': 'positive_energetic',
    'excitement': 'positive_energetic',
    'amusement': 'positive_energetic',
    'pride': 'positive_energetic',  # Added pride (similar to awe)
    
    # Positive calm
    'contentment': 'positive_calm',
    'satisfaction': 'positive_calm',
    'relief': 'positive_calm',  # Added relief
    
    # Negative intense
    'anger': 'negative_intense',
    'fear': 'negative_intense',
    'disgust': 'negative_intense',
    
    # Negative subdued
    'sadness': 'negative_subdued',
    'embarrassment': 'negative_subdued',
    'shame': 'negative_subdued',  # Added shame
    'guilt': 'negative_subdued',   # Added guilt
    
    # Surprise
    'surprise': 'surprise',
    
    # Contempt
    'contempt': 'contempt'
}

# Map 6 groups to 3 simple groups
GROUPS_6TO3 = {
    'positive_energetic': 'positive',
    'positive_calm': 'positive',
    'negative_intense': 'negative',
    'negative_subdued': 'negative',
    'surprise': 'neutral',
    'contempt': 'neutral'
}

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_directory_structure():
    """Create all necessary directories"""
    print("\n" + "="*70)
    print("📁 CREATING DIRECTORY STRUCTURE")
    print("="*70)
    
    dirs_to_create = [
        RESEARCH_DIR,
        RESEARCH_DIR / "csvs",
        RESEARCH_DIR / "models",
        RESEARCH_DIR / "results"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {dir_path}")
    
    return RESEARCH_DIR / "csvs"

def load_and_clean_data():
    """Load original CSV and clean data"""
    print("\n" + "="*70)
    print("📊 LOADING ORIGINAL DATA")
    print("="*70)
    
    df = pd.read_csv(ORIGINAL_CSV)
    print(f"   Original samples: {len(df)}")
    
    # Use primary_emotion column
    print(f"\n   Primary emotion distribution:")
    for emotion, count in df['primary_emotion'].value_counts().head(15).items():
        print(f"      {emotion}: {count}")
    
    # Add 6-group mapping
    df['emotion_group'] = df['primary_emotion'].map(EMOTION_TO_6GROUPS)
    
    # Add 3-group mapping
    df['emotion_3groups'] = df['emotion_group'].map(GROUPS_6TO3)
    
    # Remove samples without group mapping (low-sample emotions)
    df_clean = df[df['emotion_group'].notna()].copy()
    
    print(f"\n   After filtering: {len(df_clean)} samples")
    print(f"   Removed: {len(df) - len(df_clean)} samples (low-count emotions)")
    
    return df_clean

def create_train_val_test_splits(df):
    """Create stratified train/val/test splits"""
    print("\n" + "="*70)
    print("🔀 CREATING TRAIN/VAL/TEST SPLITS")
    print("="*70)
    
    # Stratify by emotion_group (6 groups)
    train_val, test = train_test_split(
        df, 
        test_size=0.15, 
        stratify=df['emotion_group'],
        random_state=42
    )
    
    train, val = train_test_split(
        train_val,
        test_size=0.176,  # 0.15 / (1 - 0.15) ≈ 0.176 to get 15% of total
        stratify=train_val['emotion_group'],
        random_state=42
    )
    
    print(f"   Train: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print(f"\n   6-Group distribution in each split:")
    for group in sorted(df['emotion_group'].unique()):
        train_pct = (train['emotion_group'] == group).sum() / len(train) * 100
        val_pct = (val['emotion_group'] == group).sum() / len(val) * 100
        test_pct = (test['emotion_group'] == group).sum() / len(test) * 100
        print(f"      {group:20s}: Train={train_pct:5.1f}% Val={val_pct:5.1f}% Test={test_pct:5.1f}%")
    
    return train, val, test

def save_experiment_csvs(train, val, test, csv_dir):
    """Save CSV files for all experiments"""
    
    experiments = {
        # Experiment 1: All emotions with sufficient data
        'all_emotions': {
            'label_col': 'primary_emotion',
            'filter': lambda df: df  # No filtering
        },
        
        # Experiment 2: 13 filtered emotions
        '13_emotions': {
            'label_col': 'primary_emotion',
            'filter': lambda df: df[df['primary_emotion'].isin(FILTERED_EMOTIONS)]
        },
        
        # Experiment 3: 6 hierarchical groups (YOUR MAIN CONTRIBUTION)
        '6_groups': {
            'label_col': 'emotion_group',
            'filter': lambda df: df  # Already filtered
        },
        
        # Experiment 4: 3 simple groups
        '3_groups': {
            'label_col': 'emotion_3groups',
            'filter': lambda df: df  # Already has 3-group mapping
        }
    }
    
    print("\n" + "="*70)
    print("💾 SAVING EXPERIMENT CSV FILES")
    print("="*70)
    
    for exp_name, config in experiments.items():
        print(f"\n   📁 {exp_name}:")
        
        for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
            # Apply filter
            filtered_df = config['filter'](split_df.copy())
            
            # Create output dataframe with required columns
            output_df = pd.DataFrame({
                'gif_id': filtered_df['gif_id'],
                'emotion_label': filtered_df[config['label_col']],
                'gif_path': filtered_df['gif_path']
            })
            
            # Save
            output_file = csv_dir / f"{split_name}_{exp_name}.csv"
            output_df.to_csv(output_file, index=False)
            
            print(f"      ✅ {output_file.name}: {len(output_df)} samples")

def verify_setup(csv_dir):
    """Verify all files are created correctly"""
    print("\n" + "="*70)
    print("✅ VERIFICATION")
    print("="*70)
    
    experiments = ['all_emotions', '13_emotions', '6_groups', '3_groups']
    splits = ['train', 'val', 'test']
    
    all_good = True
    
    for exp in experiments:
        print(f"\n   {exp}:")
        for split in splits:
            csv_file = csv_dir / f"{split}_{exp}.csv"
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                
                # Check columns
                required_cols = ['gif_id', 'emotion_label']
                has_cols = all(col in df.columns for col in required_cols)
                
                # Count unique emotions
                n_emotions = df['emotion_label'].nunique()
                
                status = "✅" if has_cols else "⚠️"
                print(f"      {status} {split}_{exp}.csv: {len(df)} samples, {n_emotions} emotions")
                
                if not has_cols:
                    print(f"         Missing columns: {set(required_cols) - set(df.columns)}")
                    all_good = False
            else:
                print(f"      ❌ {split}_{exp}.csv: NOT FOUND")
                all_good = False
    
    return all_good

def create_summary_report(train, val, test, csv_dir):
    """Create a summary report"""
    print("\n" + "="*70)
    print("📊 GENERATING SUMMARY REPORT")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("RESEARCH_TEST SETUP SUMMARY")
    report.append("="*70)
    report.append("")
    report.append(f"Setup Date: {pd.Timestamp.now()}")
    report.append(f"Base Directory: {BASE_DIR}")
    report.append(f"Research Directory: {RESEARCH_DIR}")
    report.append(f"GIF Directory: {GIF_DIR}")
    report.append("")
    
    report.append("="*70)
    report.append("DATASET STATISTICS")
    report.append("="*70)
    report.append(f"Total samples: {len(train) + len(val) + len(test)}")
    report.append(f"  Train: {len(train)} ({len(train)/(len(train)+len(val)+len(test))*100:.1f}%)")
    report.append(f"  Val:   {len(val)} ({len(val)/(len(train)+len(val)+len(test))*100:.1f}%)")
    report.append(f"  Test:  {len(test)} ({len(test)/(len(train)+len(val)+len(test))*100:.1f}%)")
    report.append("")
    
    report.append("="*70)
    report.append("EXPERIMENT 3: 6 HIERARCHICAL GROUPS (MAIN CONTRIBUTION)")
    report.append("="*70)
    
    full_df = pd.concat([train, val, test])
    for group in sorted(full_df['emotion_group'].unique()):
        count = (full_df['emotion_group'] == group).sum()
        pct = count / len(full_df) * 100
        report.append(f"  {group:25s}: {count:4d} ({pct:5.1f}%)")
    
    # Calculate imbalance ratio
    counts = full_df['emotion_group'].value_counts()
    imbalance = counts.max() / counts.min()
    report.append(f"\n  Class imbalance ratio: {imbalance:.2f}:1")
    report.append("")
    
    report.append("="*70)
    report.append("EXPERIMENT FILES CREATED")
    report.append("="*70)
    report.append("  1. all_emotions: All emotions from original dataset")
    report.append("  2. 13_emotions: 13 emotions with >= 150 samples")
    report.append("  3. 6_groups: 6 hierarchical emotion groups (YOUR MAIN RESULT)")
    report.append("  4. 3_groups: 3 simple groups (positive/negative/neutral)")
    report.append("")
    
    report.append("="*70)
    report.append("NEXT STEPS")
    report.append("="*70)
    report.append("1. Update experiment_configs.py with these paths:")
    report.append(f"   DATA_DIR = '{csv_dir}'")
    report.append(f"   GIF_DIR = '{GIF_DIR}'")
    report.append("")
    report.append("2. Run your first experiment:")
    report.append("   python train_ablation.py \\")
    report.append("       --experiment exp3_6groups \\")
    report.append(f"       --gif_dir '{GIF_DIR}' \\")
    report.append(f"       --data_dir '{csv_dir}' \\")
    report.append("       --epochs 20")
    report.append("")
    report.append("="*70)
    
    # Print to console
    for line in report:
        print(line)
    
    # Save to file
    report_file = RESEARCH_DIR / "SETUP_SUMMARY.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n💾 Report saved to: {report_file}")

def main():
    """Main setup function"""
    print("="*70)
    print("🚀 RESEARCH_TEST SETUP SCRIPT")
    print("="*70)
    print(f"Setting up: {RESEARCH_DIR}")
    print("")
    
    # Step 1: Create directories
    csv_dir = create_directory_structure()
    
    # Step 2: Load and clean data
    df = load_and_clean_data()
    
    # Step 3: Create splits
    train, val, test = create_train_val_test_splits(df)
    
    # Step 4: Save experiment CSVs
    save_experiment_csvs(train, val, test, csv_dir)
    
    # Step 5: Verify
    all_good = verify_setup(csv_dir)
    
    # Step 6: Create summary
    create_summary_report(train, val, test, csv_dir)
    
    # Final message
    print("\n" + "="*70)
    if all_good:
        print("✅ SETUP COMPLETE - ALL FILES READY!")
    else:
        print("⚠️  SETUP COMPLETE - SOME ISSUES DETECTED")
        print("   Review the verification output above")
    print("="*70)
    print("")
    print("📁 Directory structure:")
    print(f"   {RESEARCH_DIR}/")
    print(f"   ├── csvs/           (All experiment CSV files)")
    print(f"   ├── models/         (Model checkpoints will be saved here)")
    print(f"   ├── results/        (Results will be saved here)")
    print(f"   └── SETUP_SUMMARY.txt")
    print("")
    print("🚀 Ready to start training!")
    print("   Run: python train_ablation.py --experiment exp3_6groups \\")
    print(f"            --gif_dir '{GIF_DIR}' \\")
    print(f"            --data_dir '{csv_dir}'")

if __name__ == "__main__":
    main()
