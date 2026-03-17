import { useState, useRef } from 'react';
import { Upload, Loader2, Copy, RefreshCw, Home, Star, ChevronDown, Sparkles, AlertTriangle } from 'lucide-react';
import { generateCaption } from '../services/captionService';

interface MainAppProps {
  onBackToHome: () => void;
}

const SAMPLE_GIFS = [
  {
    id: '1',
    name: 'Laughing',
    url: 'https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXRtMGk1cmxnZTFka284bDRpcWdub2RzZDJxd2d5Z3N2bGtsZDRsdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/CoDp6NnSmItoY/giphy.gif'
  },
  {
    id: '2',
    name: 'Crying',
    url: 'https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjh0cGV3a2phZWQ0bmk0M2VvMnZnZmM3cWdpYjZ0ZnBkcDN5MWlvMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/HoSyEAe48WBpTCmEz4/giphy.gif'
  },
  {
    id: '3',
    name: 'Yawning',
    url: 'https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2d5YW9ycmlldDBhbnJpemhzcjAzYnJ4Z2R4M2pqbWcxYnhueDdrMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l1J9RqOoi2UMxsfzq/giphy.gif'
  },
  {
    id: '4',
    name: 'Confused',
    url: 'https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNjJ4OXBndnRrdzBqYzVlZjI1b2c2M2w4OWVyNTcwcmk5dzJ6bzk1ZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/nTfdeBvfgzV26zjoFP/giphy.gif'
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
  const [showSampleDropdown, setShowSampleDropdown] = useState(false);
  const [copiedText, setCopiedText] = useState(false);
  const [contentWarning, setContentWarning] = useState<string>(''); // NEW: Content warning state
  const [confidence, setConfidence] = useState<number>(0); // NEW: Confidence state
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
    if (!uploadedFile && !previewUrl) {
      alert('Please upload a GIF or enter a GIF URL');
      return;
    }

    setLoading(true);
    setEmotion('');
    setCaption('');
    setContentWarning(''); // Clear previous warning
    setConfidence(0);

    try {
      // Generate caption using your backend
      const result = await generateCaption(uploadedFile || previewUrl);
      
      // Update UI with results
      setEmotion(result.emotion);
      setCaption(result.caption);
      setContentWarning(result.content_warning || ''); // NEW: Capture warning
      setConfidence(result.confidence || 0); // NEW: Capture confidence
      
      // Show the results view
      setShowResults(true);
      
      // Optional: Log for debugging
      console.log('Caption generated:', result);
      
    } catch (err) {
      console.error('Error generating caption:', err);
      alert('Failed to generate caption. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRegenerate = async () => {
    setLoading(true);
    try {
      const inputForModel = uploadedFile || previewUrl;
      const result = await generateCaption(inputForModel);

      // Optional delay for UI feel
      await new Promise(resolve => setTimeout(resolve, 800));

      setCaption(result.caption);
      setEmotion(result.emotion);
      setContentWarning(result.content_warning || '');
      setConfidence(result.confidence || 0);
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

  const handleRating = (value: number) => {
    setRating(value);
    // Ratings are local state only (no Supabase persistence).
  };

  const handleUploadAnother = () => {
    setUploadedFile(null);
    setPreviewUrl('');
    setSelectedSample('');
    setCaption('');
    setEmotion('');
    setShowResults(false);
    setRating(0);
    setContentWarning('');
    setConfidence(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-purple-900 via-pink-800 to-purple-900 flex flex-col">
      {/* Background Decor */}
      <div className="fixed inset-0 opacity-20 pointer-events-none">
        <div className="absolute top-40 left-20 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
        <div className="absolute top-60 right-20 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-700"></div>
        <div className="absolute -bottom-20 left-1/2 w-96 h-96 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col max-w-7xl mx-auto w-full px-4 sm:px-6 py-6 sm:py-8">
        {/* Header */}
        <div className="text-center mb-6 sm:mb-8">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-space-grotesk font-bold bg-gradient-to-r from-pink-200 via-purple-200 to-pink-200 bg-clip-text text-transparent mb-3 sm:mb-4 tracking-tight">
            SENTIVUE
          </h1>
          <p className="text-purple-100 text-base sm:text-lg font-poppins px-4">
            Generate emotion-aware captions for your GIFs
          </p>
        </div>

        {!showResults ? (
          /* Upload View */
          <div className="flex-1 max-w-2xl mx-auto w-full space-y-6 sm:space-y-8">
            {previewUrl && (
              <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl shadow-xl p-4 sm:p-6 border border-pink-300/30 backdrop-blur-xl">
                <h3 className="text-lg font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4">
                  Preview
                </h3>
                <div className="aspect-video bg-gradient-to-br from-pink-400/20 to-purple-400/20 rounded-xl overflow-hidden border border-pink-300/30">
                  <img
                    src={previewUrl}
                    alt="GIF Preview"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            )}

            <div className="mb-6 sm:mb-8">
              <label className="block text-base sm:text-lg font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-3 sm:mb-4">
                Try Sample GIFs
              </label>
              <div className="relative">
                <button
                  onClick={() => setShowSampleDropdown(!showSampleDropdown)}
                  className="w-full bg-gradient-to-r from-pink-500/20 to-purple-500/20 border-2 border-pink-300/50 rounded-xl px-3 sm:px-4 py-2 sm:py-3 flex items-center justify-between hover:from-pink-500/30 hover:to-purple-500/30 hover:border-pink-300 transition-all duration-300 group text-sm sm:text-base"
                >
                  <span className="text-purple-100 font-poppins group-hover:text-white transition-colors">
                    {selectedSample || 'Select a sample GIF'}
                  </span>
                  <ChevronDown className={`w-4 h-4 sm:w-5 sm:h-5 text-pink-300 transition-transform duration-300 ${showSampleDropdown ? 'rotate-180' : ''}`} />
                </button>
                {showSampleDropdown && (
                  <div className="absolute z-20 w-full mt-2 bg-gradient-to-br from-purple-900/95 to-pink-900/95 backdrop-blur-xl border border-pink-300/50 rounded-xl overflow-hidden shadow-2xl">
                    {SAMPLE_GIFS.map((sample) => (
                      <button
                        key={sample.id}
                        onClick={() => handleSampleSelect(sample)}
                        className="w-full px-3 sm:px-4 py-2 sm:py-3 text-left text-purple-100 hover:bg-pink-500/20 hover:text-white transition-all duration-200 font-poppins border-b border-pink-300/20 last:border-0 text-sm sm:text-base"
                      >
                        {sample.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="mb-6 sm:mb-8">
              <label className="block text-base sm:text-lg font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-3 sm:mb-4">
                Or Upload Your GIF
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
                  className="w-full bg-gradient-to-r from-pink-500/20 to-purple-500/20 border-2 border-pink-300/50 rounded-xl px-3 sm:px-4 py-2 sm:py-3 flex items-center gap-2 sm:gap-3 cursor-pointer hover:from-pink-500/30 hover:to-purple-500/30 hover:border-pink-300 transition-all duration-300 group text-xs sm:text-sm"
                >
                  <div className="bg-gradient-to-r from-pink-500 to-purple-500 text-white px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg font-medium font-poppins flex items-center gap-2 group-hover:shadow-lg group-hover:shadow-pink-500/50 transition-all duration-300 transform group-hover:scale-105 whitespace-nowrap">
                    <Upload className="w-3 h-3 sm:w-4 sm:h-4" />
                    Choose File
                  </div>
                  <span className="text-purple-100 font-poppins group-hover:text-white transition-colors truncate">
                    {uploadedFile?.name || 'No file chosen'}
                  </span>
                </label>
              </div>
            </div>

            <div className="flex justify-center pb-4">
              <button
                onClick={handleGenerateCaption}
                disabled={!previewUrl || loading}
                className="relative group px-8 sm:px-10 py-2.5 sm:py-3 text-base sm:text-lg font-semibold text-white font-poppins overflow-hidden rounded-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-pink-500 via-purple-500 to-pink-500 transition-all duration-300 group-hover:scale-110 group-disabled:scale-100"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-pink-600 via-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative block transition-all duration-300 group-hover:scale-110 flex items-center gap-2 justify-center group-disabled:scale-100">
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 sm:w-5 sm:h-5" />
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
          <div className="flex-1 space-y-6 sm:space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8">
              <div className="md:col-span-1">
                <h2 className="text-lg sm:text-xl font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4">
                  Uploaded GIF
                </h2>
                <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl shadow-xl p-4 sm:p-6 border border-pink-300/30 backdrop-blur-xl hover:border-pink-300/60 hover:shadow-2xl hover:shadow-pink-500/30 transition-all duration-500 group cursor-pointer transform hover:scale-105">
                  <div className="aspect-square bg-gradient-to-br from-pink-400/20 to-purple-400/20 rounded-xl overflow-hidden border border-pink-300/30 group-hover:border-pink-300 transition-all duration-300">
                    <img
                      src={previewUrl}
                      alt="Uploaded GIF"
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                    />
                  </div>
                </div>
              </div>

              <div className="md:col-span-2 space-y-4 sm:space-y-6">
                {/* Content Warning Display - NEW */}
                {contentWarning && (
                  <div className="bg-yellow-500/20 border-2 border-yellow-500/50 rounded-2xl p-4 sm:p-6 backdrop-blur-sm hover:border-yellow-500/70 transition-all duration-300 group">
                    <div className="flex items-start gap-3 sm:gap-4">
                      <div className="flex-shrink-0">
                        <AlertTriangle className="w-6 h-6 sm:w-8 sm:h-8 text-yellow-300 group-hover:scale-110 transition-transform duration-300" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-sm sm:text-lg font-bold font-space-grotesk text-yellow-100 mb-2">
                          Content Warning
                        </h3>
                        <p className="text-yellow-200/90 text-xs sm:text-sm font-poppins mb-3">
                          {contentWarning}
                        </p>
                        <div className="bg-yellow-400/10 border border-yellow-400/30 rounded-lg p-2 sm:p-3">
                          <p className="text-yellow-200/80 text-xs font-poppins">
                            💡 <strong>Tip:</strong> For best results, use real-world GIFs with:
                          </p>
                          <ul className="text-yellow-200/70 text-xs mt-2 space-y-1 ml-4 font-poppins">
                            <li>• Visible human faces or clear animals</li>
                            <li>• Good lighting and clear expressions</li>
                            <li>• Recognizable actions or movements</li>
                          </ul>
                        </div>
                        {confidence > 0 && (
                          <p className="text-yellow-200/60 text-xs mt-3 font-poppins">
                            Detection confidence: {(confidence * 100).toFixed(1)}%
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                <div>
                  <h2 className="text-lg sm:text-xl font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4">
                    Generated Caption
                  </h2>
                  <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl shadow-xl p-4 sm:p-8 border border-pink-300/30 backdrop-blur-xl hover:from-white/15 hover:to-pink-500/15 hover:border-pink-300/60 transition-all duration-500 group">
                    <div className="mb-4 inline-flex flex-wrap gap-2">
                      <span className="bg-gradient-to-r from-pink-500/50 to-purple-500/50 text-white px-3 sm:px-4 py-1.5 sm:py-2 rounded-full text-xs sm:text-sm font-bold capitalize font-poppins border border-pink-300/50 group-hover:border-pink-300 group-hover:shadow-lg group-hover:shadow-pink-500/50 transition-all duration-300">
                        {emotion.replace('_', ' ')}
                      </span>
                      {confidence > 0 && (
                        <span className="text-purple-200/70 text-xs sm:text-sm font-poppins py-1.5 sm:py-2">
                          {(confidence * 100).toFixed(1)}% confident
                        </span>
                      )}
                    </div>
                    <p className="text-purple-50 text-base sm:text-lg leading-relaxed mb-4 sm:mb-6 font-poppins group-hover:text-white transition-colors duration-300">
                      "{caption}"
                    </p>

                    <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
                      <button
                        onClick={handleRegenerate}
                        disabled={loading}
                        className="flex items-center justify-center gap-2 px-4 sm:px-6 py-2 sm:py-3 bg-gradient-to-r from-pink-500/30 to-purple-500/30 border border-pink-300/50 hover:from-pink-500/50 hover:to-purple-500/50 hover:border-pink-300 text-white font-semibold font-poppins rounded-xl transition-all duration-300 disabled:opacity-50 transform hover:scale-105 hover:shadow-lg hover:shadow-pink-500/30 text-sm sm:text-base"
                      >
                        <RefreshCw className={`w-3 h-3 sm:w-4 sm:h-4 ${loading ? 'animate-spin' : ''}`} />
                        Regenerate
                      </button>
                      <button
                        onClick={handleCopy}
                        className="flex items-center justify-center gap-2 px-4 sm:px-6 py-2 sm:py-3 bg-gradient-to-r from-pink-500/30 to-purple-500/30 border border-pink-300/50 hover:from-pink-500/50 hover:to-purple-500/50 hover:border-pink-300 text-white font-semibold font-poppins rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-lg hover:shadow-pink-500/30 text-sm sm:text-base"
                      >
                        <Copy className="w-3 h-3 sm:w-4 sm:h-4" />
                        {copiedText ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl shadow-xl p-4 sm:p-8 border border-pink-300/30 backdrop-blur-xl hover:from-white/15 hover:to-pink-500/15 hover:border-pink-300/60 transition-all duration-500 group">
                    <h3 className="text-base sm:text-lg font-space-grotesk font-bold bg-gradient-to-r from-pink-200 to-purple-200 bg-clip-text text-transparent mb-4 group-hover:from-pink-100 group-hover:to-purple-100 transition-all duration-300">
                      How well does this caption match the emotion of your GIF?
                    </h3>
                    <div className="flex gap-2 sm:gap-3 justify-center">
                      {[1, 2, 3, 4, 5].map((value) => (
                        <button
                          key={value}
                          onClick={() => handleRating(value)}
                          onMouseEnter={() => setHoveredStar(value)}
                          onMouseLeave={() => setHoveredStar(0)}
                          className="transition-all duration-300 transform hover:scale-150 hover:-translate-y-2"
                        >
                          <Star
                            className={`w-8 h-8 sm:w-12 sm:h-12 transition-all duration-200 ${
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

            <div className="flex flex-col sm:flex-row justify-center gap-3 sm:gap-4 pb-4">
              <button
                onClick={handleUploadAnother}
                className="relative group px-6 sm:px-8 py-2.5 sm:py-3 text-base sm:text-lg font-semibold text-white font-poppins overflow-hidden rounded-full order-2 sm:order-1"
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
                className="relative group px-6 sm:px-8 py-2.5 sm:py-3 text-base sm:text-lg font-semibold text-white font-poppins overflow-hidden rounded-full order-1 sm:order-2"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500 transition-all duration-300 group-hover:scale-110"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-purple-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative block transition-all duration-300 group-hover:scale-110 flex items-center gap-2 justify-center">
                  <Home className="w-4 h-4 sm:w-5 sm:h-5" />
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
