import { ImageIcon, MessageSquare, Film } from 'lucide-react';

interface LandingPageProps {
  onGetStarted: () => void;
}

export default function LandingPage({ onGetStarted }: LandingPageProps) {
  return (
    <div className="w-full min-h-screen bg-gradient-to-br from-purple-900 via-pink-800 to-purple-900 flex flex-col">
      <div className="fixed inset-0 opacity-30 pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-700"></div>
        <div className="absolute -bottom-20 left-1/2 w-72 h-72 bg-pink-400 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col max-w-6xl mx-auto w-full px-4 sm:px-6 py-8 sm:py-16">
        <div className="text-center mb-8 sm:mb-16">
          <div className="flex items-center justify-center gap-2 sm:gap-3 mb-4 sm:mb-6 flex-wrap">
            <Film className="w-6 h-6 sm:w-8 sm:h-8 text-pink-300 animate-spin flex-shrink-0" />
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-space-grotesk font-bold bg-gradient-to-r from-pink-200 via-purple-200 to-pink-200 bg-clip-text text-transparent tracking-tight">
              SENTIVUE
            </h1>
            <Film className="w-6 h-6 sm:w-8 sm:h-8 text-purple-300 animate-spin flex-shrink-0" style={{ animationDirection: 'reverse' }} />
          </div>
          <p className="text-lg sm:text-xl lg:text-2xl text-purple-100 max-w-3xl mx-auto font-poppins font-light px-2">
            Where every GIF tells a feeling, not just a story.
          </p>
        </div>

        <div className="bg-gradient-to-br from-white/10 to-purple-500/10 rounded-2xl sm:rounded-3xl shadow-2xl p-6 sm:p-12 mb-8 sm:mb-12 border border-purple-300/30 backdrop-blur-xl hover:from-white/20 hover:to-pink-500/20 transition-all duration-500">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8 items-center">
            <div className="bg-gradient-to-br from-pink-400/20 to-purple-400/20 rounded-2xl p-6 sm:p-8 aspect-video flex items-center justify-center border-2 border-pink-300/50 hover:border-pink-300 transition-all duration-300 group cursor-pointer hover:shadow-2xl hover:shadow-pink-500/50">
              <div className="text-center group-hover:scale-110 transition-transform duration-300">
                <ImageIcon className="w-16 h-16 sm:w-24 sm:h-24 text-pink-300 mx-auto mb-3 sm:mb-4 group-hover:text-pink-200 transition-colors" />
                <p className="text-purple-100 font-medium font-poppins text-sm sm:text-base">GIF Video Preview</p>
              </div>
            </div>

            <div className="flex items-center justify-center">
              <div className="bg-gradient-to-br from-pink-300/20 to-purple-300/20 rounded-2xl p-6 sm:p-8 border-2 border-pink-300/50 relative hover:border-pink-300 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/50 group w-full">
                <div className="absolute -left-3 sm:-left-4 top-1/2 transform -translate-y-1/2 w-0 h-0 border-t-6 sm:border-t-8 border-t-transparent border-b-6 sm:border-b-8 border-b-transparent border-r-6 sm:border-r-8 border-r-pink-300/50 group-hover:border-r-pink-300 transition-colors hidden sm:block"></div>

                <div className="flex items-start gap-2 sm:gap-3 mb-3 group-hover:translate-x-1 transition-transform duration-300">
                  <MessageSquare className="w-4 h-4 sm:w-5 sm:h-5 text-pink-300 flex-shrink-0 mt-1 group-hover:text-pink-200 transition-colors" />
                  <p className="text-purple-100 text-xs sm:text-sm font-medium font-poppins">Generated Caption</p>
                </div>

                <div className="space-y-2">
                  <div className="h-2 bg-gradient-to-r from-pink-300 to-purple-300 rounded w-full group-hover:w-full transition-all duration-300"></div>
                  <div className="h-2 bg-gradient-to-r from-pink-300 to-purple-300 rounded w-5/6 group-hover:w-full transition-all duration-300 delay-75"></div>
                  <div className="h-2 bg-gradient-to-r from-pink-300 to-purple-300 rounded w-4/6 group-hover:w-full transition-all duration-300 delay-150"></div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-center mt-8 sm:mt-12">
            <button
              onClick={onGetStarted}
              className="relative group px-8 sm:px-12 py-3 sm:py-4 text-base sm:text-lg font-semibold text-white font-poppins overflow-hidden rounded-full"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-pink-500 via-purple-500 to-pink-500 transition-all duration-300 group-hover:scale-110"></div>
              <div className="absolute inset-0 bg-gradient-to-r from-pink-600 via-purple-600 to-pink-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <span className="relative block transition-transform duration-300 group-hover:scale-110">
                Get Started
              </span>
              <div className="absolute inset-0 rounded-full shadow-lg shadow-pink-500/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            </button>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-2xl sm:rounded-3xl p-6 sm:p-10 shadow-2xl border border-pink-400/30 backdrop-blur-xl group hover:border-pink-400/60 transition-all duration-500 hover:shadow-2xl hover:shadow-pink-500/20">
          <div className="flex items-start gap-4 sm:gap-6">
            <span className="text-5xl sm:text-6xl text-pink-300 group-hover:text-pink-200 transition-colors duration-300 flex-shrink-0">"</span>
            <p className="text-purple-50 text-base sm:text-lg leading-relaxed pt-3 sm:pt-4 font-poppins group-hover:text-white transition-colors duration-300">
              SentiVue is an AI-powered system that automatically generates emotion-aware
              captions for GIFs. By analyzing facial expressions, objects, and actions within a GIF,
              SentiVue crafts captions that reflect the true sentiment behind each moment,
              transforming simple animations into emotionally expressive stories.
            </p>
            <span className="text-5xl sm:text-6xl text-purple-300 group-hover:text-purple-200 transition-colors duration-300 self-end flex-shrink-0">"</span>
          </div>
        </div>
      </div>
    </div>
  );
}
