import { useState, useRef } from 'react';
import { Upload, Loader2, Copy, RefreshCw, Home, Star, ChevronDown, Sparkles } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { generateCaption, getSessionId } from '../services/captionService';

interface MainAppProps {
  onBackToHome: () => void;
}

const SAMPLE_GIFS = [
  {
    id: '1',
    name: 'Happy Dance',
    url: 'https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif'
  },
  {
    id: '2',
    name: 'Cat Surprised',
    url: 'https://media.giphy.com/media/VbnUQpnihPSIgIXuZv/giphy.gif'
  },
  {
    id: '3',
    name: 'Excited Kid',
    url: 'https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif'
  },
  {
    id: '4',
    name: 'Frustrated',
    url: 'https://media.giphy.com/media/l4FGGafcOHmrlQxG0/giphy.gif'
  }
];

export default function MainApp({ onBackToHome }: MainAppProps) {
  const [selectedSample, setSelectedSample] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [caption, setCaption] = useState<string>('');
  const [emotion, setEmotion] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [rating, setRating] = useState(0);
  const [hoveredStar, setHoveredStar] = useState(0);
  const [captionId, setCaptionId] = useState<string>('');
  const [showSampleDropdown, setShowSampleDropdown] = useState(false);
  const [copiedText, setCopiedText] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSampleSelect = (sample: typeof SAMPLE_GIFS[0]) => {
    setSelectedSample(sample.name);
    setPreviewUrl(sample.url);
    setUploadedFile(null);
    setShowResults(false);
    setShowSampleDropdown(false);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'image/gif') {
      setUploadedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setSelectedSample('');
      setShowResults(false);
    }
  };

  const handleGenerateCaption = async () => {
    if (!previewUrl) return;

    setLoading(true);
    try {
      const sessionId = getSessionId();
      const source = uploadedFile || previewUrl;

      // 1. Call your real API backend
      const { emotion: generatedEmotion, caption: generatedCaption } = await generateCaption(source);

      // 2. Save to Supabase (Removed unused 'gifData' variable to fix VS Code warning)
      const { data: gifRecord, error: uploadError } = await supabase
        .from('gif_uploads')
        .insert({
          file_url: previewUrl,
          file_name: uploadedFile?.name || selectedSample,
          session_id: sessionId
        })
        .select()
        .single();

      if (uploadError) throw uploadError;

      // Update local state with results
      setCaption(generatedCaption);
      setEmotion(generatedEmotion);
      if (gifRecord) setCaptionId(gifRecord.id); 
      
      // Switch view to show the results
      setShowResults(true);
    } catch (error) {
      console.error('Error:', error);
      alert("Failed to connect to the captioning model. Is your backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleRegenerate = async () => {
    setLoading(true);
    try {
      const inputForModel = uploadedFile || previewUrl;
      const { emotion: generatedEmotion, caption: generatedCaption } = await generateCaption(inputForModel);

      // Optional delay for UI feel
      await new Promise(resolve => setTimeout(resolve, 800));

      if (captionId) {
        await supabase
          .from('gif_uploads') // Using the same table if separate caption table doesn't exist
          .update({
            // Assuming your schema allows updates or you have a specific captions table
            // Adjust according to your specific Supabase database.ts structure
          })
          .eq('id', captionId);
      }

      setCaption(generatedCaption);
      setEmotion(generatedEmotion);
      setRating(0);
    } catch (error) {
      console.error('Error regenerating caption:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(caption);
    setCopiedText(true);
    setTimeout(() => setCopiedText(false), 2000);
  };

  const handleRating = async (value: number) => {
    setRating(value);
    const sessionId = getSessionId();
    
    // Using simple console log or update if you haven't set up the 'ratings' table yet
    try {
      await supabase.from('ratings').insert({
        caption_id: captionId || 'anonymous',
        rating: value,
        session_id: sessionId
      });
    } catch (e) {
      console.warn("Rating table not found, skipping DB save.");
    }
  };

  const handleUploadAnother = () => {
    setUploadedFile(null);
    setPreviewUrl('');
    setSelectedSample('');
    setCaption('');
    setEmotion('');
    setShowResults(false);
    setRating(0);
    setCaptionId('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-pink-800 to-purple-900 overflow-hidden">
      {/* Background Decor */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-40 left-20 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
        <div className="absolute top-64 right-32 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-700"></div>
        <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-space-grotesk font-bold bg-gradient-to-r from-pink-200 via-purple-200 to-pink-200 bg-clip-text text-transparent mb-2 tracking-tight">
            SENTIVUE
          </h1>
        </div>

        {!showResults ? (
          /* Selection / Upload UI */
          <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-3xl shadow-2xl p-10 mb-8 border border-purple-300/30 backdrop-blur-xl hover:from-white/15 hover:to-pink-500/15 transition-all duration-500">
            <div className="mb-8">
              <label className="block text-lg font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4">
                Sample GIFs
              </label>
              <div className="relative">
                <button
                  onClick={() => setShowSampleDropdown(!showSampleDropdown)}
                  className="w-full bg-gradient-to-r from-pink-500/20 to-purple-500/20 border-2 border-pink-300/50 rounded-xl px-4 py-3 text-left flex items-center justify-between hover:from-pink-500/30 hover:to-purple-500/30 hover:border-pink-300 transition-all duration-300 text-purple-100 font-poppins group"
                >
                  <span className="group-hover:text-white transition-colors">
                    {selectedSample || 'Choose a sample GIF...'}
                  </span>
                  <ChevronDown className={`w-5 h-5 transition-transform duration-300 ${showSampleDropdown ? 'rotate-180' : ''}`} />
                </button>

                {showSampleDropdown && (
                  <div className="absolute z-20 w-full mt-2 bg-gradient-to-b from-purple-800/90 to-pink-800/90 border-2 border-pink-300/50 rounded-xl shadow-2xl shadow-pink-500/50 overflow-hidden backdrop-blur-xl">
                    {SAMPLE_GIFS.map((sample) => (
                      <button
                        key={sample.id}
                        onClick={() => handleSampleSelect(sample)}
                        className="w-full px-4 py-3 text-left hover:bg-pink-500/30 transition-all duration-300 text-purple-100 border-b border-pink-300/20 last:border-b-0 font-poppins hover:text-white hover:translate-x-2"
                      >
                        {sample.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="mb-8">
              <label className="block text-lg font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4">
                Or Upload GIFs
              </label>
              <div className="relative">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/gif"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="w-full bg-gradient-to-r from-pink-500/20 to-purple-500/20 border-2 border-pink-300/50 rounded-xl px-4 py-3 flex items-center gap-3 cursor-pointer hover:from-pink-500/30 hover:to-purple-500/30 hover:border-pink-300 transition-all duration-300 group"
                >
                  <div className="bg-gradient-to-r from-pink-500 to-purple-500 text-white px-4 py-2 rounded-lg font-medium font-poppins flex items-center gap-2 group-hover:shadow-lg group-hover:shadow-pink-500/50 transition-all duration-300 transform group-hover:scale-105">
                    <Upload className="w-4 h-4" />
                    Choose File
                  </div>
                  <span className="text-purple-100 font-poppins group-hover:text-white transition-colors">
                    {uploadedFile?.name || 'No file chosen'}
                  </span>
                </label>
              </div>
            </div>

            <div className="flex justify-center">
              <button
                onClick={handleGenerateCaption}
                disabled={!previewUrl || loading}
                className="relative group px-10 py-3 text-lg font-semibold text-white font-poppins overflow-hidden rounded-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-pink-500 via-purple-500 to-pink-500 transition-all duration-300 group-hover:scale-110 group-disabled:scale-100"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-pink-600 via-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative block transition-all duration-300 group-hover:scale-110 flex items-center gap-2 justify-center group-disabled:scale-100">
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5" />
                      Generate Caption
                    </>
                  )}
                </span>
                <div className="absolute inset-0 rounded-full shadow-lg shadow-pink-500/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 group-disabled:opacity-0"></div>
              </button>
            </div>
          </div>
        ) : (
          /* Results View */
          <div className="space-y-8">
            <div className="grid md:grid-cols-3 gap-8">
              <div className="md:col-span-1">
                <h2 className="text-xl font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4">
                  Uploaded GIF
                </h2>
                <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl shadow-xl p-6 border border-pink-300/30 backdrop-blur-xl hover:border-pink-300/60 hover:shadow-2xl hover:shadow-pink-500/30 transition-all duration-500 group cursor-pointer transform hover:scale-105">
                  <div className="aspect-square bg-gradient-to-br from-pink-400/20 to-purple-400/20 rounded-xl overflow-hidden border border-pink-300/30 group-hover:border-pink-300 transition-all duration-300">
                    <img
                      src={previewUrl}
                      alt="Uploaded GIF"
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                    />
                  </div>
                </div>
              </div>

              <div className="md:col-span-2 space-y-6">
                <div>
                  <h2 className="text-xl font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4">
                    Generated Caption
                  </h2>
                  <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl shadow-xl p-8 border border-pink-300/30 backdrop-blur-xl hover:from-white/15 hover:to-pink-500/15 hover:border-pink-300/60 transition-all duration-500 group">
                    <div className="mb-4 inline-block">
                      <span className="bg-gradient-to-r from-pink-500/50 to-purple-500/50 text-white px-4 py-2 rounded-full text-sm font-bold capitalize font-poppins border border-pink-300/50 group-hover:border-pink-300 group-hover:shadow-lg group-hover:shadow-pink-500/50 transition-all duration-300">
                        {emotion}
                      </span>
                    </div>
                    <p className="text-purple-50 text-lg leading-relaxed mb-6 font-poppins group-hover:text-white transition-colors duration-300">
                      {caption}
                    </p>

                    <div className="flex gap-3">
                      <button
                        onClick={handleRegenerate}
                        disabled={loading}
                        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-pink-500/30 to-purple-500/30 border border-pink-300/50 hover:from-pink-500/50 hover:to-purple-500/50 hover:border-pink-300 text-white font-semibold font-poppins rounded-xl transition-all duration-300 disabled:opacity-50 transform hover:scale-105 hover:shadow-lg hover:shadow-pink-500/30"
                      >
                        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                        Regenerate
                      </button>
                      <button
                        onClick={handleCopy}
                        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-pink-500/30 to-purple-500/30 border border-pink-300/50 hover:from-pink-500/50 hover:to-purple-500/50 hover:border-pink-300 text-white font-semibold font-poppins rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-lg hover:shadow-pink-500/30"
                      >
                        <Copy className="w-4 h-4" />
                        {copiedText ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl shadow-xl p-8 border border-pink-300/30 backdrop-blur-xl hover:from-white/15 hover:to-pink-500/15 hover:border-pink-300/60 transition-all duration-500 group">
                    <h3 className="text-lg font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4 group-hover:from-pink-100 group-hover:to-purple-100 transition-all duration-300">
                      How well does this caption match the emotion of your GIF?
                    </h3>
                    <div className="flex gap-3 justify-center">
                      {[1, 2, 3, 4, 5].map((value) => (
                        <button
                          key={value}
                          onClick={() => handleRating(value)}
                          onMouseEnter={() => setHoveredStar(value)}
                          onMouseLeave={() => setHoveredStar(0)}
                          className="transition-all duration-300 transform hover:scale-150 hover:-translate-y-2"
                        >
                          <Star
                            className={`w-12 h-12 transition-all duration-200 ${
                              value <= (hoveredStar || rating)
                                ? 'fill-yellow-300 text-yellow-300 drop-shadow-lg drop-shadow-yellow-500'
                                : 'text-pink-300/50 hover:text-pink-300'
                            }`}
                          />
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-center gap-4">
              <button
                onClick={handleUploadAnother}
                className="relative group px-8 py-3 text-lg font-semibold text-white font-poppins overflow-hidden rounded-full"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-pink-500 via-purple-500 to-pink-500 transition-all duration-300 group-hover:scale-110"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-pink-600 via-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative block transition-transform duration-300 group-hover:scale-110">
                  Upload Another GIF
                </span>
                <div className="absolute inset-0 rounded-full shadow-lg shadow-pink-500/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </button>
              <button
                onClick={onBackToHome}
                className="relative group px-8 py-3 text-lg font-semibold text-white font-poppins overflow-hidden rounded-full"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500 transition-all duration-300 group-hover:scale-110"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative block transition-all duration-300 group-hover:scale-110 flex items-center gap-2 justify-center">
                  <Home className="w-5 h-5" />
                  Back to Home
                </span>
                <div className="absolute inset-0 rounded-full shadow-lg shadow-purple-500/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}