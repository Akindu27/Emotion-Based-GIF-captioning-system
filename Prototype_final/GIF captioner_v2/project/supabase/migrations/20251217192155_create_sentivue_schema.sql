/*
  # SentiVue GIF Captioning System Schema

  1. New Tables
    - `gif_uploads`
      - `id` (uuid, primary key) - Unique identifier for each upload
      - `file_url` (text) - Storage URL for the uploaded GIF
      - `file_name` (text) - Original filename
      - `uploaded_at` (timestamptz) - Upload timestamp
      - `session_id` (text) - Anonymous session identifier
    
    - `captions`
      - `id` (uuid, primary key) - Unique identifier for each caption
      - `gif_id` (uuid, foreign key) - Reference to gif_uploads
      - `caption_text` (text) - Generated caption content
      - `emotion` (text) - Detected emotion (happy, sad, excited, etc.)
      - `generated_at` (timestamptz) - Generation timestamp
    
    - `ratings`
      - `id` (uuid, primary key) - Unique identifier for each rating
      - `caption_id` (uuid, foreign key) - Reference to captions
      - `rating` (integer) - Star rating (1-5)
      - `created_at` (timestamptz) - Rating timestamp
      - `session_id` (text) - Anonymous session identifier

  2. Security
    - Enable RLS on all tables
    - Public read/write access for demo purposes (no authentication required)
    - Policies allow anonymous users to interact with the system
*/

-- Create gif_uploads table
CREATE TABLE IF NOT EXISTS gif_uploads (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  file_url text NOT NULL,
  file_name text NOT NULL,
  uploaded_at timestamptz DEFAULT now(),
  session_id text NOT NULL
);

-- Create captions table
CREATE TABLE IF NOT EXISTS captions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  gif_id uuid NOT NULL REFERENCES gif_uploads(id) ON DELETE CASCADE,
  caption_text text NOT NULL,
  emotion text NOT NULL,
  generated_at timestamptz DEFAULT now()
);

-- Create ratings table
CREATE TABLE IF NOT EXISTS ratings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  caption_id uuid NOT NULL REFERENCES captions(id) ON DELETE CASCADE,
  rating integer NOT NULL CHECK (rating >= 1 AND rating <= 5),
  created_at timestamptz DEFAULT now(),
  session_id text NOT NULL
);

-- Enable Row Level Security
ALTER TABLE gif_uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE captions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ratings ENABLE ROW LEVEL SECURITY;

-- Policies for gif_uploads (allow public access for demo)
CREATE POLICY "Anyone can upload GIFs"
  ON gif_uploads FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Anyone can view GIF uploads"
  ON gif_uploads FOR SELECT
  TO anon
  USING (true);

-- Policies for captions (allow public access)
CREATE POLICY "Anyone can create captions"
  ON captions FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Anyone can view captions"
  ON captions FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Anyone can update captions"
  ON captions FOR UPDATE
  TO anon
  USING (true)
  WITH CHECK (true);

-- Policies for ratings (allow public access)
CREATE POLICY "Anyone can submit ratings"
  ON ratings FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Anyone can view ratings"
  ON ratings FOR SELECT
  TO anon
  USING (true);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_captions_gif_id ON captions(gif_id);
CREATE INDEX IF NOT EXISTS idx_ratings_caption_id ON ratings(caption_id);
CREATE INDEX IF NOT EXISTS idx_gif_uploads_session_id ON gif_uploads(session_id);