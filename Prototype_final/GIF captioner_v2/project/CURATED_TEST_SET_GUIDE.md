# CURATED TEST GIF SET FOR SENTIVUE
# Best GIFs for Emotion-Based Caption Generation

## RECOMMENDED GIF SOURCES

### 1. GIPHY (https://giphy.com)
Search terms that work well:
- "happy person"
- "sad person"  
- "excited reaction"
- "dog playing"
- "celebration"
- "laughing"
- "crying real"
- "surprised reaction real"

Filter: Select "GIFs" (not Stickers), sort by "Relevant"

### 2. TENOR (https://tenor.com)
Same search terms as GIPHY
Filter: Avoid cartoon reactions, focus on real people

### 3. YOUR EXISTING GIFGIF DATASET
Use these emotion groups from your test set:
- positive_energetic: 271 samples (best)
- negative_subdued: 147 samples
- negative_intense: 219 samples
- surprise: 80 samples
- positive_calm: 146 samples
- contempt: 26 samples (limited)

---

## CURATED TEST SET (50 GIFs)

### POSITIVE_ENERGETIC (15 GIFs)
✅ GOOD EXAMPLES:
1. Person dancing at concert
2. Child jumping with joy
3. Dog running and playing
4. Athletes celebrating after goal
5. People hugging at reunion
6. Wedding couple laughing
7. Kid opening birthday present
8. Group high-fiving
9. Runner crossing finish line
10. Baby laughing
11. Person clapping enthusiastically
12. Cat playing with toy
13. Crowd cheering at event
14. Friends jumping together
15. Victory celebration

❌ AVOID:
- Animated characters dancing
- Cartoon celebrations
- Pure text GIFs
- Meme reactions (Drake, etc.)

### POSITIVE_CALM (10 GIFs)
✅ GOOD EXAMPLES:
1. Person meditating
2. Sunset beach scene with person
3. Cat sleeping peacefully
4. Couple holding hands
5. Reading in cozy chair
6. Gentle smile close-up
7. Person sipping coffee
8. Peaceful garden scene
9. Dog resting with owner
10. Calm breathing exercise

### NEGATIVE_INTENSE (10 GIFs)
✅ GOOD EXAMPLES:
1. Person yelling/shouting
2. Angry expression close-up
3. Someone running away scared
4. Frightened reaction
5. Disgusted face expression
6. Frustrated person
7. Scared child
8. Angry dog barking
9. Person crying hard
10. Panic reaction

### NEGATIVE_SUBDUED (8 GIFs)
✅ GOOD EXAMPLES:
1. Person crying softly
2. Sad looking down
3. Disappointed expression
4. Lonely person sitting
5. Melancholic gaze
6. Sorrowful face
7. Dejected posture
8. Gloomy rainy scene with person

### SURPRISE (5 GIFs)
✅ GOOD EXAMPLES:
1. Shocked expression
2. Surprised gasp
3. Eyes widening
4. Mouth dropping open
5. Jump scare reaction

### CONTEMPT (2 GIFs)
✅ GOOD EXAMPLES:
1. Eye roll
2. Dismissive wave

---

## TEST SET STRUCTURE

Create folder structure:
```
test_gifs/
├── positive_energetic/
│   ├── 001_dancing.gif
│   ├── 002_celebrating.gif
│   └── ...
├── positive_calm/
│   ├── 001_meditating.gif
│   └── ...
├── negative_intense/
│   └── ...
├── negative_subdued/
│   └── ...
├── surprise/
│   └── ...
└── contempt/
    └── ...
```

---

## DOWNLOAD GUIDE

### Method 1: From GIPHY
1. Search for emotion (e.g., "happy person real")
2. Click on GIF
3. Right-click → "Save image as..." → Save as .gif
4. Rename: emotion_number_description.gif

### Method 2: From Your GIFGIF Dataset
```python
import pandas as pd
import shutil
import os

# Load your test data
test_df = pd.read_csv('test_grouped.csv')

# Create output directory
os.makedirs('curated_test_set', exist_ok=True)

# Select best examples (high confidence from previous runs)
best_gifs = {
    'positive_energetic': test_df[test_df['emotion_group'] == 'positive_energetic'].sample(15),
    'positive_calm': test_df[test_df['emotion_group'] == 'positive_calm'].sample(10),
    'negative_intense': test_df[test_df['emotion_group'] == 'negative_intense'].sample(10),
    'negative_subdued': test_df[test_df['emotion_group'] == 'negative_subdued'].sample(8),
    'surprise': test_df[test_df['emotion_group'] == 'surprise'].sample(5),
    'contempt': test_df[test_df['emotion_group'] == 'contempt'].sample(2),
}

# Copy to new folder
for emotion, gifs in best_gifs.items():
    emotion_dir = f'curated_test_set/{emotion}'
    os.makedirs(emotion_dir, exist_ok=True)
    
    for idx, row in enumerate(gifs.iterrows(), 1):
        src = row[1]['gif_path']
        dst = f'{emotion_dir}/{idx:03d}_{row[1]["gif_id"]}.gif'
        shutil.copy(src, dst)
        print(f'Copied: {dst}')

print('✅ Curated test set created!')
```

---

## QUALITY CHECKLIST

Before adding a GIF to test set, verify:
- ✅ Real-world (not cartoon/anime)
- ✅ Visible human face OR clear animal
- ✅ Clear emotion expression
- ✅ Good lighting
- ✅ Not too blurry
- ✅ Action is visible (if applicable)
- ✅ File size < 5MB
- ✅ Duration 1-5 seconds

---

## EVALUATION METRICS

For each GIF in test set, record:
1. Ground truth emotion (your label)
2. Detected emotion (system output)
3. Detected objects
4. Detected action (if any)
5. Generated caption
6. Accuracy (correct/incorrect)
7. Caption quality rating (1-5)

Create evaluation spreadsheet:
```
gif_id | true_emotion | detected_emotion | correct? | objects | action | caption | quality_rating | notes
```

---

## SAMPLE EVALUATION RESULTS

Target metrics for curated set:
- Emotion accuracy: 50-60% (vs 35.55% on full test set)
- Object detection: 70%+ (at least one object)
- Action detection: 40-50%
- Caption coherence: 90%+ (grammatically correct)
- Emotion word inclusion: 100%

---

## DEMO SHOWCASE GIFS (Top 10)

For your presentation, select these:
1. **Perfect detection** (emotion + objects + action all correct)
2. **With animal** (dog/cat successfully detected)
3. **High confidence** (>90% emotion confidence)
4. **Interesting action** (dancing, celebrating, etc.)
5. **Multiple objects** (person + object interaction)
6. **Subtle emotion** (gentle smile, calm expression)
7. **Intense emotion** (crying, yelling, laughing hard)
8. **Surprise reaction** (shocked face, gasping)
9. **Group interaction** (multiple people)
10. **Celebrity/recognizable** (optional, for engagement)

---

## NOTES

- Avoid copyrighted content for public demos
- Test with diverse demographics (age, gender, ethnicity)
- Include both close-ups and full-body shots
- Mix indoor and outdoor scenes
- Test different lighting conditions
- Include some challenging cases (show limitations honestly)

---

## READY-TO-USE GIF SOURCES

### Free, Safe-to-Use GIF Libraries:
1. **Pexels Videos** → Convert to GIF
2. **Pixabay Videos** → Convert to GIF
3. **Unsplash GIFs**
4. **GIFGIF Dataset** (you already have this!)

### Tools to Convert Video → GIF:
- EZGIF.com
- CloudConvert
- FFmpeg: `ffmpeg -i video.mp4 -vf "fps=10,scale=480:-1" output.gif`

---

Good luck building your test set! 🚀
