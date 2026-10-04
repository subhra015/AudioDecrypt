"use client";

import { useState, useRef, useEffect } from "react";
import { blobToWav } from "@/lib/audio";
import { 
  Mic, 
  MicOff, 
  Loader2, 
  CheckCircle, 
  XCircle,
  Activity,
  TrendingUp,
  AlertTriangle,
  Radio
} from "lucide-react";

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{
    isSpoof: boolean;
    confidence: number;
    realProb: number;
    fakeProb: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detectionHistory, setDetectionHistory] = useState<Array<{
    time: string;
    confidence: number;
    isFake: boolean;
  }>>([]);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const animationRef = useRef<number | null>(null);
  const [waveformData, setWaveformData] = useState<number[]>(new Array(100).fill(0));
  const [frequencyData, setFrequencyData] = useState<number[]>(new Array(50).fill(0));

  // Real-time audio visualization
  useEffect(() => {
    if (isRecording) {
      const setupAudioVisualization = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
          analyserRef.current = audioContextRef.current.createAnalyser();
          analyserRef.current.fftSize = 256;
          
          const source = audioContextRef.current.createMediaStreamSource(stream);
          source.connect(analyserRef.current);
          
          const bufferLength = analyserRef.current.frequencyBinCount;
          dataArrayRef.current = new Uint8Array(bufferLength);
          
          const updateVisualization = () => {
            if (!analyserRef.current || !dataArrayRef.current) return;
            
            // Get waveform data
            analyserRef.current.getByteTimeDomainData(dataArrayRef.current);
            const waveform = Array.from(dataArrayRef.current).map(v => (v - 128) / 128.0);
            setWaveformData(waveform.slice(0, 100));
            
            // Get frequency data for heatmap
            analyserRef.current.getByteFrequencyData(dataArrayRef.current);
            const frequencies = Array.from(dataArrayRef.current).slice(0, 50);
            setFrequencyData(frequencies);
            
            animationRef.current = requestAnimationFrame(updateVisualization);
          };
          
          updateVisualization();
        } catch (err) {
          console.error("Visualization error:", err);
        }
      };
      
      setupAudioVisualization();
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      setWaveformData(new Array(100).fill(0));
      setFrequencyData(new Array(50).fill(0));
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, [isRecording]);

  const startRecording = async () => {
    setError(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError("Microphone access denied. Please allow permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      mediaRecorderRef.current.onstop = async () => {
        const webmBlob = new Blob(chunksRef.current, { type: "audio/webm" });
        await processAudio(webmBlob);
      };
    }
  };

  const processAudio = async (webmBlob: Blob) => {
    setIsProcessing(true);
    setError(null);
    
    try {
      const wavBlob = await blobToWav(webmBlob);
      const formData = new FormData();
      formData.append("file", wavBlob, "recording.wav");

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Backend processing failed");
      }

      const data = await response.json();
      const isSpoof = data.prediction === "fake";
      
      const newResult = {
        isSpoof,
        confidence: data.confidence,
        realProb: data.real_probability,
        fakeProb: data.fake_probability
      };
      
      setResult(newResult);
      
      setDetectionHistory(prev => [...prev.slice(-9), {
        time: new Date().toLocaleTimeString(),
        confidence: data.confidence,
        isFake: isSpoof
      }]);
      
    } catch (err: any) {
      console.error("Fetch Error:", err);
      setError(err.message || "Failed to connect to backend.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Generate heatmap colors based on frequency intensity
  const getHeatmapColor = (value: number, index: number, total: number) => {
    const normalizedIndex = index / total;
    const intensity = value / 255;
    
    // AI voices typically have different patterns in mid-high frequencies
    if (result?.isSpoof) {
      // Red/Purple gradient for AI
      const r = Math.floor(200 + 55 * intensity);
      const b = Math.floor(100 + 155 * (1 - normalizedIndex));
      const g = Math.floor(50 * (1 - intensity));
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Green/Blue gradient for real
      const g = Math.floor(150 + 105 * intensity);
      const b = Math.floor(200 + 55 * (1 - normalizedIndex));
      const r = Math.floor(50 * (1 - intensity));
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            AI Voice Detection Radar
          </h1>
          <p className="text-gray-400">Real-time Deepfake Detection System</p>
        </div>

        {/* Main Control Panel */}
        <div className="bg-gray-800 rounded-2xl p-8 shadow-2xl border border-gray-700">
          <div className="flex flex-col items-center gap-6">
            {!isRecording ? (
              <button
                onClick={startRecording}
                disabled={isProcessing}
                className="flex items-center gap-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-8 rounded-full transition-all transform hover:scale-105 shadow-lg"
              >
                <Mic size={24} /> Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="flex items-center gap-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-bold py-4 px-8 rounded-full transition-all animate-pulse shadow-lg"
              >
                <MicOff size={24} /> Stop Recording
              </button>
            )}

            {result && (
              <div className={`flex flex-col items-center gap-4 p-8 rounded-2xl w-full max-w-md border-2 ${
                result.isSpoof 
                  ? "bg-red-950/50 border-red-500 shadow-red-500/50 shadow-xl" 
                  : "bg-green-950/50 border-green-500 shadow-green-500/50 shadow-xl"
              }`}>
                {result.isSpoof ? (
                  <XCircle size={64} className="text-red-500 animate-pulse" />
                ) : (
                  <CheckCircle size={64} className="text-green-500" />
                )}
                <span className={`text-3xl font-bold ${
                  result.isSpoof ? "text-red-400" : "text-green-400"
                }`}>
                  {result.isSpoof ? "️ AI/SPOOF DETECTED" : "✓ Authentic Voice"}
                </span>
                <div className="flex gap-6 text-lg">
                  <span className="text-blue-400">Real: {(result.realProb * 100).toFixed(1)}%</span>
                  <span className="text-purple-400">Fake: {(result.fakeProb * 100).toFixed(1)}%</span>
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 text-red-400 bg-red-900/20 p-4 rounded-lg w-full max-w-md border border-red-500/30">
                <AlertTriangle size={20} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Real-time Waveform & Frequency Heatmap */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Waveform */}
          <div className="bg-gray-800 rounded-2xl p-6 shadow-xl border border-gray-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2 text-blue-400">
              <Activity />
              Real-time Waveform
            </h2>
            <div className="h-40 bg-gray-900 rounded-lg overflow-hidden relative">
              {isRecording ? (
                <svg className="w-full h-full" preserveAspectRatio="none">
                  <polyline
                    points={waveformData.map((v, i) => `${(i / 100) * 100}%,${50 + v * 40}%`).join(' ')}
                    fill="none"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="2"
                  />
                </svg>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500 text-sm">
                  Press "Start Recording" to see waveform
                </div>
              )}
            </div>
          </div>

          {/* Frequency Heatmap */}
          <div className="bg-gray-800 rounded-2xl p-6 shadow-xl border border-gray-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2 text-purple-400">
              <Radio />
              Frequency Spectrum Analysis
            </h2>
            <div className="h-40 bg-gray-900 rounded-lg overflow-hidden relative">
              {isRecording ? (
                <div className="flex h-full items-end gap-1 p-2">
                  {frequencyData.map((value, index) => (
                    <div
                      key={index}
                      className="flex-1 rounded-t transition-all duration-75"
                      style={{
                        height: `${(value / 255) * 100}%`,
                        background: result 
                          ? getHeatmapColor(value, index, frequencyData.length)
                          : `hsl(${200 + (index / frequencyData.length) * 60}, 70%, 50%)`,
                        opacity: 0.8 + (value / 255) * 0.2
                      }}
                      title={`Freq ${index}: ${value}`}
                    />
                  ))}
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500 text-sm">
                  Press "Start Recording" to see spectrum
                </div>
              )}
            </div>
            <div className="mt-2 flex justify-between text-xs text-gray-400">
              <span>Low Freq (0-500Hz)</span>
              <span>Mid Freq (500-2000Hz)</span>
              <span>High Freq (2000+Hz)</span>
            </div>
          </div>
        </div>

        {/* AI Detection Heatmap Grid */}
        {result && (
          <div className="bg-gray-800 rounded-2xl p-6 shadow-xl border border-gray-700">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <TrendingUp className={result.isSpoof ? "text-red-400" : "text-green-400"} />
              AI Detection Heatmap - Frequency Band Analysis
            </h2>
            
            <div className="grid md:grid-cols-4 gap-4 mb-6">
              {/* Low Frequency Band */}
              <div className={`p-4 rounded-xl border-2 ${
                result.isSpoof ? "bg-red-900/30 border-red-500/50" : "bg-green-900/30 border-green-500/50"
              }`}>
                <div className="text-sm text-gray-400 mb-2">Low Frequencies</div>
                <div className="text-2xl font-bold mb-1">0-500 Hz</div>
                <div className="text-xs text-gray-300">
                  {result.isSpoof ? "⚠️ Unnatural harmonics" : "✓ Natural pitch"}
                </div>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${result.isSpoof ? "bg-red-500" : "bg-green-500"}`}
                    style={{ width: `${result.realProb * 100}%` }}
                  />
                </div>
              </div>

              {/* Mid-Low Frequency Band */}
              <div className={`p-4 rounded-xl border-2 ${
                result.isSpoof ? "bg-orange-900/30 border-orange-500/50" : "bg-green-900/30 border-green-500/50"
              }`}>
                <div className="text-sm text-gray-400 mb-2">Mid-Low</div>
                <div className="text-2xl font-bold mb-1">500-1500 Hz</div>
                <div className="text-xs text-gray-300">
                  {result.isSpoof ? "⚠️ Formant irregularities" : "✓ Natural formants"}
                </div>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${result.isSpoof ? "bg-orange-500" : "bg-green-500"}`}
                    style={{ width: `${result.realProb * 100}%` }}
                  />
                </div>
              </div>

              {/* Mid-High Frequency Band */}
              <div className={`p-4 rounded-xl border-2 ${
                result.isSpoof ? "bg-yellow-900/30 border-yellow-500/50" : "bg-green-900/30 border-green-500/50"
              }`}>
                <div className="text-sm text-gray-400 mb-2">Mid-High</div>
                <div className="text-2xl font-bold mb-1">1500-4000 Hz</div>
                <div className="text-xs text-gray-300">
                  {result.isSpoof ? "⚠️ Spectral artifacts" : "✓ Clean spectrum"}
                </div>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${result.isSpoof ? "bg-yellow-500" : "bg-green-500"}`}
                    style={{ width: `${result.realProb * 100}%` }}
                  />
                </div>
              </div>

              {/* High Frequency Band */}
              <div className={`p-4 rounded-xl border-2 ${
                result.isSpoof ? "bg-purple-900/30 border-purple-500/50" : "bg-green-900/30 border-green-500/50"
              }`}>
                <div className="text-sm text-gray-400 mb-2">High Frequencies</div>
                <div className="text-2xl font-bold mb-1">4000+ Hz</div>
                <div className="text-xs text-gray-300">
                  {result.isSpoof ? "⚠️ Neural vocoder traces" : "✓ Natural noise"}
                </div>
                <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${result.isSpoof ? "bg-purple-500" : "bg-green-500"}`}
                    style={{ width: `${result.realProb * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Overall Risk Assessment */}
            <div className={`p-6 rounded-xl border-2 text-center ${
              result.isSpoof 
                ? "bg-gradient-to-r from-red-900/50 to-red-800/30 border-red-500" 
                : "bg-gradient-to-r from-green-900/50 to-green-800/30 border-green-500"
            }`}>
              <div className="text-sm text-gray-300 mb-2">OVERALL RISK ASSESSMENT</div>
              <div className={`text-5xl font-bold mb-2 ${
                result.isSpoof ? "text-red-400" : "text-green-400"
              }`}>
                {result.isSpoof ? "HIGH RISK - AI GENERATED" : "LOW RISK - AUTHENTIC"}
              </div>
              <div className="text-xl">
                Confidence: <span className="font-bold">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Detection History */}
        {detectionHistory.length > 0 && (
          <div className="bg-gray-800 rounded-2xl p-6 shadow-xl border border-gray-700">
            <h2 className="text-xl font-bold mb-4">Detection History</h2>
            <div className="space-y-2">
              {detectionHistory.map((detection, idx) => (
                <div 
                  key={idx} 
                  className={`flex justify-between items-center p-4 rounded-lg border ${
                    detection.isFake 
                      ? "bg-red-900/20 border-red-500/30" 
                      : "bg-green-900/20 border-green-500/30"
                  }`}
                >
                  <span className="text-gray-300 font-mono">{detection.time}</span>
                  <span className={`font-bold flex items-center gap-2 ${
                    detection.isFake ? "text-red-400" : "text-green-400"
                  }`}>
                    {detection.isFake ? <XCircle size={16} /> : <CheckCircle size={16} />}
                    {detection.isFake ? "AI Detected" : "Authentic"}
                  </span>
                  <span className="text-blue-400 font-bold">{(detection.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}