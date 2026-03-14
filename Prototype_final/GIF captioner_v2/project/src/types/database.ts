export interface GifUpload {
  id: string;
  file_url: string;
  file_name: string;
  uploaded_at: string;
  session_id: string;
}

export interface Caption {
  id: string;
  gif_id: string;
  caption_text: string;
  emotion: string;
  generated_at: string;
}

export interface Rating {
  id: string;
  caption_id: string;
  rating: number;
  created_at: string;
  session_id: string;
}
