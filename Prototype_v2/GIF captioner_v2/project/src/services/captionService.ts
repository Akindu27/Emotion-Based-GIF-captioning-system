// services/captionService.ts

const API_URL = "http://127.0.0.1:8000";

export async function generateCaption(fileOrUrl: File | string): Promise<{ emotion: string; caption: string }> {
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
  const response = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to generate caption");
  }

  // 3. Parse the JSON from main.py
  return await response.json(); 
}

// Keep this for Supabase tracking
export function getSessionId(): string {
  let sessionId = localStorage.getItem('sentivue_session_id');
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
    localStorage.setItem('sentivue_session_id', sessionId);
  }
  return sessionId;
}