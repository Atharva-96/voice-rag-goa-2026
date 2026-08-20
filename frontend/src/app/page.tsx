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
    try {
      const res = await fetch(`${BACKEND_URL}/api/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (e) {
      console.error("Failed to fetch historical metrics", e);
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
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Title & SEO Meta */}
      <title>Svara RAG | Voice-Enabled RAG Dashboard</title>

      {/* Top Navigation / Status Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <span className="font-extrabold text-sm text-white">स्वर</span>
          </div>
          <div>
            <h1 id="app-title" className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              Svara RAG
            </h1>
            <p className="text-xs text-slate-500">Hacker House Goa 2026</p>
          </div>
        </div>

        {/* Cold Start Indicator & Warmup trigger */}
        <div className="flex items-center gap-3">
          {isWarmingUp ? (
            <div className="flex items-center gap-2 text-yellow-400 text-xs bg-yellow-400/10 px-3 py-1.5 rounded-full border border-yellow-400/20">
              <span className="w-2 h-2 rounded-full bg-yellow-400 animate-ping" />
              <span>Backend Warming Up (Render Cold Start)...</span>
            </div>
          ) : isWarmedUp ? (
            <div className="flex items-center gap-2 text-emerald-400 text-xs bg-emerald-400/10 px-3 py-1.5 rounded-full border border-emerald-400/20">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span>Backend Active</span>
            </div>
          ) : (
            <button 
              type="button"
              onClick={checkWarmupStatus}
              className="text-xs text-slate-300 bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-full border border-slate-700 transition-all flex items-center gap-2"
            >
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span>Check Server State</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 grid-rows-none lg:grid-cols-12 gap-6">
        {/* Left Column - User Controls */}
        <section className="lg:col-span-5 flex flex-col gap-6">
          
          {/* Glassmorphic Recorder Card */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-lg flex flex-col items-center justify-center text-center gap-6 shadow-xl relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent pointer-events-none" />
            
            <div>
              <h2 className="text-lg font-semibold text-slate-200">Voice Input Query</h2>
              <p className="text-xs text-slate-500 mt-1">Record your question in Hindi or English (Sarvam Saaras v3)</p>
            </div>

            {/* Glowing Record Button */}
            <div className="relative flex items-center justify-center">
              {isRecording && (
                <div className="absolute w-36 h-36 rounded-full border-2 border-indigo-500/30 animate-ping pointer-events-none" />
              )}
              <button
                id="voice-record-btn"
                type="button"
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isQuerying}
                className={`relative w-28 h-28 rounded-full flex flex-col items-center justify-center shadow-2xl transition-all duration-300 ${
                  isRecording
                    ? "bg-gradient-to-br from-rose-500 to-red-600 ring-4 ring-rose-500/20 hover:scale-95"
                    : "bg-gradient-to-br from-indigo-600 to-violet-700 ring-4 ring-indigo-500/20 hover:scale-105 hover:shadow-indigo-500/10"
                } disabled:opacity-50`}
              >
                {isRecording ? (
                  <>
                    <div className="w-6 h-6 bg-white rounded-sm animate-pulse" />
                    <span className="text-[10px] text-white font-medium mt-2">STOP</span>
                  </>
                ) : (
                  <>
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    <span className="text-[10px] text-white font-medium mt-2">TAP TO SPEAK</span>
                  </>
                )}
              </button>
            </div>

            {isRecording ? (
              <div className="text-sm font-semibold text-rose-400 animate-pulse">
                Recording... {recordingDuration}s
              </div>
            ) : isQuerying ? (
              <div className="flex items-center gap-2 text-indigo-400 text-sm">
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Processing pipeline...</span>
              </div>
            ) : (
              <p className="text-xs text-slate-600">Supports WAV format downsampled to 16kHz mono PCM</p>
            )}

            {error && (
              <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg p-3 text-xs w-full">
                {error}
              </div>
            )}
          </div>

          {/* Text Input Form Fallback */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-lg flex flex-col gap-4 shadow-xl">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Manual Query</h2>
              <p className="text-xs text-slate-500 mt-0.5">Type your question directly for testing</p>
            </div>
            <form onSubmit={submitTextQuery} className="flex gap-2">
              <input
                id="query-text-input"
                type="text"
                placeholder="उदा. भारत की राजधानी क्या है?"
                value={typedQuery}
                onChange={(e) => setTypedQuery(e.target.value)}
                disabled={isQuerying}
                className="flex-1 bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl px-4 py-2 text-sm text-slate-200 outline-none transition-all placeholder:text-slate-700"
              />
              <button
                type="submit"
                disabled={isQuerying || !typedQuery.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl px-4 py-2 text-sm font-semibold transition-all shadow-lg shadow-indigo-600/15"
              >
                Send
              </button>
            </form>
          </div>

          {/* Guidelines / Helper Prompts */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-lg flex flex-col gap-3 shadow-xl">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Try these queries</h3>
            <div className="flex flex-col gap-2">
              <button 
                type="button"
                onClick={() => { setTypedQuery("भारत की राजधानी क्या है?"); }}
                className="text-left text-xs text-indigo-400 hover:text-indigo-300 bg-indigo-500/5 hover:bg-indigo-500/10 border border-indigo-500/10 rounded-lg px-3 py-2 transition-all"
              >
                • भारत की राजधानी क्या है? (Answers from MSMARCO Context)
              </button>
              <button 
                type="button"
                onClick={() => { setTypedQuery("कॉर्पोरेशन क्या है?"); }}
                className="text-left text-xs text-indigo-400 hover:text-indigo-300 bg-indigo-500/5 hover:bg-indigo-500/10 border border-indigo-500/10 rounded-lg px-3 py-2 transition-all"
              >
                • कॉर्पोरेशन क्या है? (Answers from MSMARCO Context)
              </button>
              <button 
                type="button"
                onClick={() => { setTypedQuery("how to hack a computer"); }}
                className="text-left text-xs text-rose-400 hover:text-rose-300 bg-rose-500/5 hover:bg-rose-500/10 border border-rose-500/10 rounded-lg px-3 py-2 transition-all"
              >
                • how to hack a computer (Triggers L1 Input Guardrail refusal)
              </button>
              <button 
                type="button"
                onClick={() => { setTypedQuery("पिज्जा बनाने की विधि क्या है?"); }}
                className="text-left text-xs text-amber-400 hover:text-amber-300 bg-amber-500/5 hover:bg-amber-500/10 border border-amber-500/10 rounded-lg px-3 py-2 transition-all"
              >
                • पिज्जा बनाने की विधि क्या है? (Triggers L2 Context score refusal)
              </button>
            </div>
          </div>

        </section>

        {/* Right Column - Results Display */}
        <section className="lg:col-span-7 flex flex-col gap-6">

          {/* Query Transcript & Answer display */}
          <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-lg flex flex-col gap-5 shadow-xl min-h-[300px]">
            <div>
              <h2 className="text-lg font-semibold text-slate-200">Response Console</h2>
              {response && (
                <p className="text-[10px] text-slate-500 font-mono mt-1">Request ID: {response.request_id}</p>
              )}
            </div>

            {response ? (
              <div className="flex flex-col gap-5">
                {/* Transcript */}
                <div>
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Detected Transcript</h3>
                  <p className="text-sm font-medium text-indigo-300 mt-1 bg-indigo-500/5 border border-indigo-500/10 rounded-xl px-4 py-3">
                    {response.query || "No query detected"}
                  </p>
                </div>

                {/* Grounded Answer */}
                <div>
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Grounded Answer</h3>
                  
                  {response.guardrail_refusal ? (
                    <div className="mt-2 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-xl p-4 flex gap-3">
                      <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      <div>
                        <p className="text-sm font-semibold">Guardrail Refusal Triggered</p>
                        <p className="text-xs mt-1 text-rose-400/80">{response.answer}</p>
                      </div>
                    </div>
                  ) : (
                    <p className="text-base font-semibold text-slate-100 mt-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-4 shadow-inner leading-relaxed">
                      {response.answer}
                    </p>
                  )}
                </div>

                {/* Stage Latencies */}
                <div>
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Query Latency Breakdown</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    
                    {response.latency.stt_ms !== null && (
                      <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-center">
                        <p className="text-[10px] text-slate-500 uppercase tracking-wider">STT (Audio)</p>
                        <p className="text-lg font-bold text-indigo-400 mt-1">{response.latency.stt_ms.toFixed(0)}<span className="text-xs font-medium ml-0.5">ms</span></p>
                      </div>
                    )}
                    
                    <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-center">
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider">Retrieval</p>
                      <p className="text-lg font-bold text-emerald-400 mt-1">{response.latency.retrieval_ms.toFixed(0)}<span className="text-xs font-medium ml-0.5">ms</span></p>
                    </div>

                    <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-center">
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider">LLM Gen</p>
                      <p className="text-lg font-bold text-amber-400 mt-1">{response.latency.llm_ms.toFixed(0)}<span className="text-xs font-medium ml-0.5">ms</span></p>
                    </div>

                    <div className="bg-slate-950 border border-indigo-950 rounded-xl p-3 text-center ring-2 ring-indigo-500/10">
                      <p className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Total Pipeline</p>
                      <p className={`text-lg font-black mt-1 ${response.latency.total_ms < 200 ? 'text-indigo-400' : 'text-slate-200'}`}>
                        {response.latency.total_ms.toFixed(0)}
                        <span className="text-xs font-medium ml-0.5">ms</span>
                      </p>
                    </div>

                  </div>
                </div>

              </div>
            ) : isQuerying ? (
              <div className="flex-1 flex flex-col items-center justify-center gap-3">
                <div className="w-10 h-10 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-sm text-slate-500 animate-pulse">Running voice/text RAG flow...</p>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 py-12">
                <svg className="w-12 h-12 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-sm text-slate-500 font-medium">Console Idle</p>
                <p className="text-xs text-slate-600 max-w-sm">Type a query or tap the microphone to run a RAG pipeline and review structured outputs.</p>
              </div>
            )}
          </div>

          {/* Sources and Context Display */}
          {response && response.sources && response.sources.length > 0 && (
            <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-lg flex flex-col gap-4 shadow-xl">
              <div>
                <h3 className="text-sm font-semibold text-slate-200">Retrieved Context Passages ({response.sources.length})</h3>
                <p className="text-xs text-slate-500">Multilingual vector retrieval matching paraphrase-multilingual</p>
              </div>
              <div className="flex flex-col gap-3">
                {response.sources.map((src, i) => (
                  <div key={src.passage_id || i} className="bg-slate-950 border border-slate-800/80 hover:border-slate-800 rounded-xl p-4 transition-all">
                    <div className="flex justify-between items-center gap-4 mb-2">
                      <span className="text-[10px] bg-slate-900 text-slate-400 font-mono px-2 py-0.5 rounded-full border border-slate-800">
                        Doc ID: {src.passage_id.substring(0, 8)}...
                      </span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        src.score >= 0.5 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/10 text-amber-400'
                      }`}>
                        Cosine Score: {src.score.toFixed(4)}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">{src.passage}</p>
                    
                    {src.metadata && src.metadata.is_selected === 1 && (
                      <div className="mt-2 flex items-center gap-1.5 text-[10px] text-emerald-400 font-medium bg-emerald-500/5 px-2 py-1 rounded-md border border-emerald-500/10 w-fit">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
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
      <footer className="border-t border-slate-800 bg-slate-900/40 backdrop-blur-md px-6 py-8 mt-12">
        <div className="max-w-7xl mx-auto flex flex-col gap-6">
          <div className="flex justify-between items-center flex-wrap gap-4">
            <div>
              <h2 className="text-lg font-bold text-slate-200">Real-time Performance Metrics</h2>
              <p className="text-xs text-slate-500">Instrumented P50 / P70 / P100 latency percentiles calculated from actual queries</p>
            </div>
            <button 
              type="button"
              onClick={fetchMetrics}
              className="text-xs font-semibold text-indigo-400 bg-indigo-500/5 hover:bg-indigo-500/10 border border-indigo-500/20 hover:border-indigo-500/30 rounded-xl px-4 py-2 transition-all flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 8H18.2" />
              </svg>
              <span>Refresh Metrics</span>
            </button>
          </div>

          {metrics && metrics.count > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
              
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Total Pipeline</p>
                <div className="mt-3 flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P50 (Median)</span><span className="font-bold text-indigo-400">{metrics.total.p50} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P70</span><span className="font-bold text-slate-300">{metrics.total.p70} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P100 (Max)</span><span className="font-bold text-slate-300">{metrics.total.p100} ms</span></div>
                </div>
              </div>

              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Vector Retrieval</p>
                <div className="mt-3 flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P50 (Median)</span><span className="font-bold text-emerald-400">{metrics.retrieval.p50} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P70</span><span className="font-bold text-slate-300">{metrics.retrieval.p70} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P100 (Max)</span><span className="font-bold text-slate-300">{metrics.retrieval.p100} ms</span></div>
                </div>
              </div>

              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">LLM Generation</p>
                <div className="mt-3 flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P50 (Median)</span><span className="font-bold text-amber-400">{metrics.llm.p50} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P70</span><span className="font-bold text-slate-300">{metrics.llm.p70} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P100 (Max)</span><span className="font-bold text-slate-300">{metrics.llm.p100} ms</span></div>
                </div>
              </div>

              <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
                <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Voice STT</p>
                <div className="mt-3 flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P50 (Median)</span><span className="font-bold text-indigo-400">{metrics.stt.p50} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P70</span><span className="font-bold text-slate-300">{metrics.stt.p70} ms</span></div>
                  <div className="flex justify-between text-xs"><span className="text-slate-400">P100 (Max)</span><span className="font-bold text-slate-300">{metrics.stt.p100} ms</span></div>
                </div>
              </div>

            </div>
          ) : (
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 text-center text-xs text-slate-600">
              No queries run yet. Latency history is blank. Performance percentiles will populate upon pipeline execution.
            </div>
          )}

          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 text-xs text-slate-600 mt-4 border-t border-slate-850 pt-4">
            <p>© 2026 Svara RAG. Built for Hacker House Goa 2026.</p>
            <div className="flex gap-4">
              <span className="text-slate-500">Target Pipeline Latency: &lt;200ms</span>
              <span>Vector Database: Qdrant Free Tier</span>
              <span>STT: Sarvam Saaras v3</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
