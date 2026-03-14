// services/captionService.ts

const API_URL = "http://127.0.0.1:8000";

// Define the response type
interface CaptionResponse {
  emotion: string;
  caption: string;
  confidence?: number;
  objects?: string[];
  content_type?: string;
  content_warning?: string;
}

export async function generateCaption(fileOrUrl: File | string): Promise<CaptionResponse> {
  const formData = new FormData();
  
  // 1. Handle Input: If it's a URL (string), fetch it. If it's a File, use it directly.
  if (typeof fileOrUrl === 'string') {
    // We add { mode: 'cors' } to ensure we can grab samples from external sites like Giphy
    const response = await fetch(fileOrUrl, { mode: 'cors' });
    const blob = await response.blob();
    formData.append('file', blob, 'sample.gif');
  } else {
    formData.append('file', fileOrUrl);
  }

  // 2. Send to your LOCAL FastAPI backend

  //const response = await fetch(`${API_URL}/generate`, {
  const response = await fetch(`${API_URL}/generate?mode=enhanced`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to generate caption");
  }

  // 3. Parse the JSON from main.py
  return await response.json(); 
}