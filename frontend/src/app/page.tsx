"use client";

import React, { useState, useEffect, useRef } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface SourceDoc {
  passage_id: string;
  passage: string;
  score: number;
  metadata: {
    query_id?: string;
    passage_index?: number;
    chunk_index?: number;
    gold_answer?: string;
    source_query?: string;
    is_selected?: number;
  };
}

interface LatencyBreakdown {
  stt_ms: number | null;
  retrieval_ms: number;
  llm_ms: number;
  total_ms: number;
}

interface QueryResponse {
  request_id: string;
  query: string;
  answer: string;
  sources: SourceDoc[];
  latency: LatencyBreakdown;
  guardrail_refusal: boolean;
}

interface Metrics {
  count: number;
  stt: { p50: number; p70: number; p100: number };
  retrieval: { p50: number; p70: number; p100: number };
  llm: { p50: number; p70: number; p100: number };
  total: { p50: number; p70: number; p100: number };
}

// Extends Window interface to avoid any casts for webkitAudioContext
interface WebkitWindow extends Window {
  AudioContext?: typeof AudioContext;
  webkitAudioContext?: typeof AudioContext;
}

export default function Home() {
  // App States
  const [isWarmedUp, setIsWarmedUp] = useState<boolean>(false);
  const [isWarmingUp, setIsWarmingUp] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [typedQuery, setTypedQuery] = useState<string>("");
  
  // Results States
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [isRefreshingMetrics, setIsRefreshingMetrics] = useState<boolean>(false);

  // Audio Recorder Refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioInputRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const recordedSamplesRef = useRef<Float32Array[]>([]);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Automatic Warmup check on page load
  useEffect(() => {
    checkWarmupStatus();
    fetchMetrics();
  }, []);

  const checkWarmupStatus = async () => {
    setIsWarmingUp(true);
    try {
      // Ping warmup endpoint. Handles Render cold starts gracefully.
      const res = await fetch(`${BACKEND_URL}/api/warmup`);
      if (res.ok) {
        setIsWarmedUp(true);
      }
    } catch (e) {
      console.error("Warmup ping failed. Server might be spun down.", e);
    } finally {
      setIsWarmingUp(false);
    }
  };

  const fetchMetrics = async () => {
    setIsRefreshingMetrics(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (e) {
      console.error("Failed to fetch historical metrics", e);
    } finally {
      // Add a small delay for better visual transition
      setTimeout(() => {
        setIsRefreshingMetrics(false);
      }, 500);
    }
  };

  // Start Audio Recording
  const startRecording = async () => {
    setError(null);
    recordedSamplesRef.current = [];
    setRecordingDuration(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // Initialize AudioContext at 16000Hz (auto downsampling)
      const w = window as unknown as WebkitWindow;
      const AudioContextClass = w.AudioContext || w.webkitAudioContext;
      if (!AudioContextClass) {
        throw new Error("Web Audio API is not supported in this browser.");
      }
      
      const audioCtx = new AudioContextClass({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      audioInputRef.current = source;

      // 4096 buffer size, 1 input channel, 1 output channel
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Clone samples buffer
        recordedSamplesRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setIsRecording(true);

      // Track Duration
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);

    } catch (err) {
      console.error("Microphone access error:", err);
      setError("Microphone permission denied or unavailable.");
    }
  };

  // Stop Audio Recording
  const stopRecording = async () => {
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
    }
    
    // Disconnect nodes
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
    }
    if (audioInputRef.current) {
      audioInputRef.current.disconnect();
    }
    if (audioContextRef.current) {
      await audioContextRef.current.close();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }

    setIsRecording(false);

    // Merge Float32Array segments
    const samples = recordedSamplesRef.current;
    if (samples.length === 0) return;

    let totalLength = 0;
    for (const arr of samples) {
      totalLength += arr.length;
    }

    const mergedBuffer = new Float32Array(totalLength);
    let offset = 0;
    for (const arr of samples) {
      mergedBuffer.set(arr, offset);
      offset += arr.length;
    }

    // Convert to WAV Blob
    const wavBlob = bufferToWav(mergedBuffer, 16000);
    submitAudioQuery(wavBlob);
  };

  // WAV Converter Helpers
  const bufferToWav = (buffer: Float32Array, sampleRate: number): Blob => {
    const bufferLength = buffer.length;
    const wavBuffer = new ArrayBuffer(44 + bufferLength * 2);
    const view = new DataView(wavBuffer);

    const writeString = (v: DataView, off: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        v.setUint8(off + i, str.charCodeAt(i));
      }
    };

    const floatTo16BitPCM = (output: DataView, off: number, input: Float32Array) => {
      for (let i = 0; i < input.length; i++, off += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
    };

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + bufferLength * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM Format
    view.setUint16(22, 1, true); // Mono Channel
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // 16-bit
    writeString(view, 36, "data");
    view.setUint32(40, bufferLength * 2, true);

    floatTo16BitPCM(view, 44, buffer);
    return new Blob([view], { type: "audio/wav" });
  };

  // Submit audio payload to backend
  const submitAudioQuery = async (audioBlob: Blob) => {
    setIsQuerying(true);
    setError(null);
    setResponse(null);

    const formData = new FormData();
    formData.append("file", audioBlob, "query.wav");

    try {
      const res = await fetch(`${BACKEND_URL}/api/query-audio`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server returned error status: ${res.status}`);
      }

      const data = await res.json() as QueryResponse;
      setResponse(data);
      fetchMetrics();
    } catch (e) {
      console.error(e);
      setError("Query failed. Ensure backend service is running and configured.");
    } finally {
      setIsQuerying(false);
    }
  };

  // Submit text query to backend
  const submitTextQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!typedQuery.trim()) return;

    setIsQuerying(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${BACKEND_URL}/api/query-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: typedQuery }),
      });

      if (!res.ok) {
        throw new Error(`Server returned error status: ${res.status}`);
      }

      const data = await res.json() as QueryResponse;
      setResponse(data);
      fetchMetrics();
      setTypedQuery("");
    } catch (e) {
      console.error(e);
      setError("Query failed. Ensure backend service is running.");
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-[#FAFAFA] flex flex-col font-sans selection:bg-[#FAFAFA] selection:text-black">
      {/* Title & SEO Meta */}
      <title>Svara RAG | Voice-Enabled RAG Dashboard</title>

      {/* Top Navigation / Status Header */}
      <header className="border-b border-[#1F1F1F] bg-black/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <h1 id="app-title" className="text-4xl font-dongle font-normal tracking-wide text-white leading-none select-none">
            Svara
          </h1>
        </div>

        {/* Cold Start Indicator & Warmup trigger */}
        <div className="flex items-center gap-3">
          {isWarmingUp ? (
            <div className="flex items-center gap-2 text-amber-400 text-xs bg-amber-500/[0.04] px-3 py-1.5 rounded-full border border-amber-500/20 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              <span>Warming Up...</span>
            </div>
          ) : isWarmedUp ? (
            <div className="flex items-center gap-2 text-emerald-400 text-xs bg-emerald-500/[0.04] px-3 py-1.5 rounded-full border border-emerald-500/20 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span>Active</span>
            </div>
          ) : (
            <button 
              type="button"
              onClick={checkWarmupStatus}
              className="text-xs text-[#A1A1A1] hover:text-[#FAFAFA] bg-transparent hover:bg-white/[0.03] px-3 py-1.5 rounded-full border border-[#1F1F1F] hover:border-[#262626] transition-all flex items-center gap-2 font-mono"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span>Check Status</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 lg:p-12 grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12">
        {/* Left Column - User Controls */}
        <section className="lg:col-span-5 flex flex-col gap-8">
          
          {/* Glassmorphic Recorder Card */}
          <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] backdrop-blur-md p-6 lg:p-8 flex flex-col items-center justify-center text-center gap-6 shadow-2xl">
            {/* Top border subtle glow */}
            <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent" />
            
            <div className="flex flex-col gap-1">
              <h2 className="text-sm font-semibold tracking-tight text-[#FAFAFA]">Voice Input Query</h2>
              <p className="text-xs text-[#A1A1A1]">Record your question in Hindi or English (Sarvam Saaras v3)</p>
            </div>

            {/* Circular Record Button */}
            <div className="relative flex items-center justify-center py-4">
              {isRecording && (
                <div className="absolute w-28 h-28 rounded-full border border-white/20 animate-ping pointer-events-none" />
              )}
              <button
                id="voice-record-btn"
                type="button"
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isQuerying}
                className={`relative w-20 h-20 rounded-full flex flex-col items-center justify-center transition-all duration-300 border focus:outline-none focus:ring-1 focus:ring-white/40 ${
                  isRecording
                    ? "bg-[#FAFAFA] border-white text-black hover:bg-[#E5E5E5] scale-95"
                    : "bg-transparent border-white/10 hover:border-white/30 text-white hover:bg-white/[0.04] scale-100"
                } disabled:opacity-30 disabled:pointer-events-none`}
              >
                {isRecording ? (
                  <div className="w-5 h-5 bg-red-600 rounded-sm animate-pulse" />
                ) : (
                  <svg className="w-6 h-6 text-[#FAFAFA] transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                )}
              </button>
            </div>

            <div className="min-h-[24px] flex items-center justify-center">
              {isRecording ? (
                <div className="text-xs font-mono font-semibold text-white tracking-widest uppercase flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  <span>RECORDING · {recordingDuration}s</span>
                </div>
              ) : isQuerying ? (
                <div className="flex items-center gap-2 text-[#A1A1A1] text-xs font-mono">
                  <span className="inline-block w-2.5 h-2.5 border border-[#A1A1A1] border-t-transparent rounded-full animate-spin" />
                  <span>PROCESSING...</span>
                </div>
              ) : (
                <p className="text-[11px] text-[#525252] font-mono uppercase tracking-wider">TAP MICROPHONE TO SPEAK</p>
              )}
            </div>

            {error && (
              <div className="bg-red-500/[0.04] border border-red-500/20 text-red-400 rounded-xl p-3 text-xs w-full text-left font-mono">
                {error}
              </div>
            )}
          </div>

          {/* Text Input Form Fallback */}
          <div className="bg-[#0F0F0F] border border-[#1F1F1F] rounded-2xl p-6 flex flex-col gap-4">
            <div>
              <h2 className="text-sm font-semibold tracking-tight text-[#FAFAFA]">Manual Query</h2>
              <p className="text-xs text-[#A1A1A1]">Type your question directly</p>
            </div>
            <form onSubmit={submitTextQuery} className="flex gap-2">
              <input
                id="query-text-input"
                type="text"
                placeholder="उदा. भारत की राजधानी क्या है?"
                value={typedQuery}
                onChange={(e) => setTypedQuery(e.target.value)}
                disabled={isQuerying}
                className="flex-1 bg-black border border-[#262626] focus:border-white focus:ring-1 focus:ring-white rounded-lg px-4 py-2 text-sm text-[#FAFAFA] outline-none transition-all placeholder:text-[#525252]"
              />
              <button
                type="submit"
                disabled={isQuerying || !typedQuery.trim()}
                className="bg-[#FAFAFA] hover:bg-[#E5E5E5] disabled:opacity-50 text-black font-medium rounded-lg px-4 py-2 text-sm transition-all focus:outline-none focus:ring-1 focus:ring-white"
              >
                Send
              </button>
            </form>
          </div>

          {/* Guidelines / Helper Prompts */}
          <div className="bg-[#0F0F0F] border border-[#1F1F1F] rounded-2xl p-6 flex flex-col gap-4">
            <h3 className="text-xs font-semibold text-[#525252] uppercase tracking-wider">Sample Queries</h3>
            <div className="flex flex-col gap-2">
              <button 
                type="button"
                onClick={() => { setTypedQuery("भारत की राजधानी क्या है?"); }}
                className="text-left text-xs text-[#A1A1A1] hover:text-[#FAFAFA] bg-transparent hover:bg-white/[0.01] border border-[#1F1F1F] hover:border-[#262626] rounded-lg px-3.5 py-3 transition-all flex items-center justify-between group"
              >
                <span>भारत की राजधानी क्या है?</span>
                <span className="text-[#525252] group-hover:text-[#FAFAFA] font-mono text-[10px]">&rarr;</span>
              </button>
              <button 
                type="button"
                onClick={() => { setTypedQuery("कॉर्पोरेशन क्या है?"); }}
                className="text-left text-xs text-[#A1A1A1] hover:text-[#FAFAFA] bg-transparent hover:bg-white/[0.01] border border-[#1F1F1F] hover:border-[#262626] rounded-lg px-3.5 py-3 transition-all flex items-center justify-between group"
              >
                <span>कॉर्पोरेशन क्या है?</span>
                <span className="text-[#525252] group-hover:text-[#FAFAFA] font-mono text-[10px]">&rarr;</span>
              </button>
              <button 
                type="button"
                onClick={() => { setTypedQuery("how to hack a computer"); }}
                className="text-left text-xs text-[#A1A1A1] hover:text-red-400 bg-transparent hover:bg-red-500/[0.02] border border-[#1F1F1F] hover:border-red-500/20 rounded-lg px-3.5 py-3 transition-all flex items-center justify-between group"
              >
                <span className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-red-500/60 animate-pulse" />
                  <span>how to hack a computer</span>
                </span>
                <span className="text-[#525252] group-hover:text-red-400 font-mono text-[10px]">&rarr;</span>
              </button>
              <button 
                type="button"
                onClick={() => { setTypedQuery("पिज्जा बनाने की विधि क्या है?"); }}
                className="text-left text-xs text-[#A1A1A1] hover:text-[#FAFAFA] bg-transparent hover:bg-white/[0.01] border border-[#1F1F1F] hover:border-[#262626] rounded-lg px-3.5 py-3 transition-all flex items-center justify-between group"
              >
                <span>पिज्जा बनाने की विधि क्या है?</span>
                <span className="text-[#525252] group-hover:text-[#FAFAFA] font-mono text-[10px]">&rarr;</span>
              </button>
            </div>
          </div>

        </section>

        {/* Right Column - Results Display */}
        <section className="lg:col-span-7 flex flex-col gap-8">

          {/* Query Transcript & Answer display */}
          <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] backdrop-blur-md p-6 lg:p-8 flex flex-col gap-6 min-h-[360px]">
            {/* Top border subtle glow */}
            <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-white/20 to-transparent" />
            
            <div className="flex justify-between items-start gap-4">
              <div>
                <h2 className="text-sm font-semibold tracking-tight text-[#FAFAFA]">Response Console</h2>
                {response && (
                  <p className="text-[10px] text-[#525252] font-mono mt-1">ID: {response.request_id}</p>
                )}
              </div>
            </div>

            {response ? (
              <div className="flex flex-col gap-6">
                {/* Transcript */}
                <div className="flex flex-col gap-2">
                  <h3 className="text-[10px] font-semibold text-[#525252] uppercase tracking-wider font-mono">Detected Transcript</h3>
                  <p className="text-sm font-medium text-[#FAFAFA] bg-black/40 border border-[#1F1F1F] rounded-xl px-4 py-3">
                    {response.query || "No query detected"}
                  </p>
                </div>

                {/* Grounded Answer */}
                <div className="flex flex-col gap-2">
                  <h3 className="text-[10px] font-semibold text-[#525252] uppercase tracking-wider font-mono">Grounded Answer</h3>
                  
                  {response.guardrail_refusal ? (
                    <div className="bg-red-500/[0.03] border border-red-500/20 text-[#FAFAFA] rounded-xl p-4 flex gap-3">
                      <svg className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      <div>
                        <p className="text-xs font-mono font-semibold text-red-500 uppercase tracking-wider">Guardrail Refusal Triggered</p>
                        <p className="text-sm mt-1 text-red-400 leading-relaxed">{response.answer}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm text-[#FAFAFA] leading-relaxed bg-black/40 border border-[#1F1F1F] rounded-xl px-4 py-4 whitespace-pre-wrap">
                      {response.answer}
                    </div>
                  )}
                </div>

                {/* Stage Latencies */}
                <div className="flex flex-col gap-2">
                  <h3 className="text-[10px] font-semibold text-[#525252] uppercase tracking-wider font-mono mb-1">Query Latency Breakdown</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    
                    {response.latency.stt_ms !== null && (
                      <div className="bg-black/40 border border-[#1F1F1F] rounded-xl p-3 flex flex-col justify-between">
                        <p className="text-[9px] text-[#525252] uppercase tracking-wider font-mono font-semibold">STT (Audio)</p>
                        <p className="text-base font-semibold font-mono text-[#FAFAFA] mt-1">{response.latency.stt_ms.toFixed(0)}<span className="text-[10px] text-[#525252] ml-0.5">ms</span></p>
                      </div>
                    )}
                    
                    <div className="bg-black/40 border border-[#1F1F1F] rounded-xl p-3 flex flex-col justify-between">
                      <p className="text-[9px] text-[#525252] uppercase tracking-wider font-mono font-semibold">Retrieval</p>
                      <p className="text-base font-semibold font-mono text-[#FAFAFA] mt-1">{response.latency.retrieval_ms.toFixed(0)}<span className="text-[10px] text-[#525252] ml-0.5">ms</span></p>
                    </div>

                    <div className="bg-black/40 border border-[#1F1F1F] rounded-xl p-3 flex flex-col justify-between">
                      <p className="text-[9px] text-[#525252] uppercase tracking-wider font-mono font-semibold">LLM Gen</p>
                      <p className="text-base font-semibold font-mono text-[#FAFAFA] mt-1">{response.latency.llm_ms.toFixed(0)}<span className="text-[10px] text-[#525252] ml-0.5">ms</span></p>
                    </div>

                    <div className="bg-black/40 border border-white/20 rounded-xl p-3 flex flex-col justify-between">
                      <p className="text-[9px] text-[#A1A1A1] uppercase tracking-wider font-mono font-semibold">Total Pipe</p>
                      <p className="text-base font-black font-mono text-white mt-1">{response.latency.total_ms.toFixed(0)}<span className="text-[10px] text-[#A1A1A1] ml-0.5">ms</span></p>
                    </div>

                  </div>
                </div>

              </div>
            ) : isQuerying ? (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 py-12">
                <div className="w-full max-w-[200px] h-[2px] bg-[#1F1F1F] rounded-full overflow-hidden relative">
                  <div className="absolute top-0 bottom-0 left-0 w-full bg-white rounded-full origin-left animate-progress" />
                </div>
                <p className="text-xs text-[#525252] font-mono uppercase tracking-widest animate-pulse">Executing RAG Pipeline...</p>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center gap-3 py-16">
                <svg className="w-8 h-8 text-[#525252]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <div className="flex flex-col gap-1">
                  <p className="text-xs text-[#A1A1A1] font-medium font-mono uppercase tracking-wider">Console Idle</p>
                  <p className="text-xs text-[#525252] max-w-xs">Run a query by manual typing or voice audio recording to populate structured outputs.</p>
                </div>
              </div>
            )}
          </div>

          {/* Sources and Context Display */}
          {response && response.sources && response.sources.length > 0 && (
            <div className="bg-[#0F0F0F] border border-[#1F1F1F] rounded-2xl p-6 flex flex-col gap-4">
              <div>
                <h3 className="text-sm font-semibold tracking-tight text-[#FAFAFA]">Retrieved Context Passages ({response.sources.length})</h3>
                <p className="text-xs text-[#A1A1A1]">Vector retrieval cosine matches</p>
              </div>
              <div className="flex flex-col gap-3">
                {response.sources.map((src, i) => (
                  <div key={src.passage_id || i} className="bg-black/30 border border-[#1F1F1F] rounded-xl p-4 flex flex-col gap-2 hover:border-[#262626] transition-all">
                    <div className="flex justify-between items-center gap-4">
                      <span className="text-[10px] bg-black text-[#525252] font-mono px-2 py-0.5 rounded border border-[#1F1F1F]">
                        ID: {src.passage_id.substring(0, 8)}...
                      </span>
                      <span className="text-[10px] font-mono text-[#A1A1A1]">
                        Cosine: {src.score.toFixed(4)}
                      </span>
                    </div>
                    <p className="text-xs text-[#A1A1A1] leading-relaxed">{src.passage}</p>
                    
                    {src.metadata && src.metadata.is_selected === 1 && (
                      <div className="flex items-center gap-1.5 text-[9px] font-mono text-emerald-400 bg-emerald-500/[0.04] px-2 py-0.5 rounded border border-emerald-500/10 w-fit">
                        <span className="w-1 h-1 rounded-full bg-emerald-400" />
                        <span>MSMARCO Gold Selected Context</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

        </section>
      </main>

      {/* Latency Instrumentation & Percentile Dashboard */}
      <footer className="border-t border-[#1F1F1F] bg-black px-6 py-8 mt-12">
        <div className="max-w-7xl mx-auto flex flex-col gap-6">
          <div className="flex justify-between items-center flex-wrap gap-4">
            <div>
              <h2 className="text-sm font-semibold tracking-tight text-[#FAFAFA]">Performance Metrics</h2>
              <p className="text-xs text-[#525252]">P50 / P70 / P100 latency percentiles calculated from actual queries</p>
            </div>
            <button 
              type="button"
              onClick={fetchMetrics}
              disabled={isRefreshingMetrics}
              className="text-xs font-medium text-[#FAFAFA] bg-transparent hover:bg-white/[0.03] disabled:opacity-50 border border-[#1F1F1F] hover:border-[#262626] rounded-lg px-3.5 py-1.5 transition-all flex items-center gap-1.5"
            >
              {isRefreshingMetrics ? (
                <svg 
                  className="w-3 h-3 animate-spin text-white" 
                  fill="none" 
                  viewBox="0 0 24 24"
                >
                  <circle 
                    className="opacity-25" 
                    cx="12" 
                    cy="12" 
                    r="10" 
                    stroke="currentColor" 
                    strokeWidth="3"
                  />
                  <path 
                    className="opacity-75" 
                    fill="currentColor" 
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
              ) : (
                <svg 
                  className="w-3.5 h-3.5" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 8H18.2" />
                </svg>
              )}
              <span>{isRefreshingMetrics ? "Refreshing..." : "Refresh"}</span>
            </button>
          </div>

          {metrics && metrics.count > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
              
              <div className="bg-black/30 border border-[#1F1F1F] rounded-xl p-4 flex flex-col gap-3">
                <p className="text-[10px] text-[#525252] uppercase tracking-wider font-semibold font-mono">Total Pipeline</p>
                <div className="flex flex-col gap-1.5 font-mono text-xs">
                  <div className="flex justify-between"><span className="text-[#525252]">P50 (Median)</span><span className="font-semibold text-white">{metrics.total.p50} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P70</span><span className="font-semibold text-[#A1A1A1]">{metrics.total.p70} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P100 (Max)</span><span className="font-semibold text-[#A1A1A1]">{metrics.total.p100} ms</span></div>
                </div>
              </div>

              <div className="bg-black/30 border border-[#1F1F1F] rounded-xl p-4 flex flex-col gap-3">
                <p className="text-[10px] text-[#525252] uppercase tracking-wider font-semibold font-mono">Vector Retrieval</p>
                <div className="flex flex-col gap-1.5 font-mono text-xs">
                  <div className="flex justify-between"><span className="text-[#525252]">P50 (Median)</span><span className="font-semibold text-white">{metrics.retrieval.p50} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P70</span><span className="font-semibold text-[#A1A1A1]">{metrics.retrieval.p70} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P100 (Max)</span><span className="font-semibold text-[#A1A1A1]">{metrics.retrieval.p100} ms</span></div>
                </div>
              </div>

              <div className="bg-black/30 border border-[#1F1F1F] rounded-xl p-4 flex flex-col gap-3">
                <p className="text-[10px] text-[#525252] uppercase tracking-wider font-semibold font-mono">LLM Generation</p>
                <div className="flex flex-col gap-1.5 font-mono text-xs">
                  <div className="flex justify-between"><span className="text-[#525252]">P50 (Median)</span><span className="font-semibold text-white">{metrics.llm.p50} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P70</span><span className="font-semibold text-[#A1A1A1]">{metrics.llm.p70} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P100 (Max)</span><span className="font-semibold text-[#A1A1A1]">{metrics.llm.p100} ms</span></div>
                </div>
              </div>

              <div className="bg-black/30 border border-[#1F1F1F] rounded-xl p-4 flex flex-col gap-3">
                <p className="text-[10px] text-[#525252] uppercase tracking-wider font-semibold font-mono">Voice STT</p>
                <div className="flex flex-col gap-1.5 font-mono text-xs">
                  <div className="flex justify-between"><span className="text-[#525252]">P50 (Median)</span><span className="font-semibold text-white">{metrics.stt.p50} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P70</span><span className="font-semibold text-[#A1A1A1]">{metrics.stt.p70} ms</span></div>
                  <div className="flex justify-between"><span className="text-[#525252]">P100 (Max)</span><span className="font-semibold text-[#A1A1A1]">{metrics.stt.p100} ms</span></div>
                </div>
              </div>

            </div>
          ) : (
            <div className="bg-black border border-[#1F1F1F] rounded-xl p-6 text-center text-xs text-[#525252] font-mono">
              NO PIPELINE RUNS DETECTED YET. PERFORMANCE PERCENTILES WILL POPULATE ON EXECUTION.
            </div>
          )}

          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 text-[11px] text-[#525252] mt-4 border-t border-[#1F1F1F] pt-6">
            <p>Built by Zero_Day_Devs · Hacker House Goa 2026</p>
            <div className="flex gap-4 font-mono text-[10px]">
              <span>STT: Sarvam Saaras v3</span>
              <span>Vector DB: Qdrant</span>
              <span>Target Pipeline Latency: &lt;200ms</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

