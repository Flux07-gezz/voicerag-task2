import React, { useState, useRef, useEffect } from 'react';

const WS_URL = 'ws://localhost:8000/ws/rag';

export default function AudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [latency, setLatency] = useState(null);
  const [guardrailTriggered, setGuardrailTriggered] = useState(false);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const recognitionRef = useRef(null);

  useEffect(() => {
    connectWebSocket();

    // Initialize Web Speech API for low-latency client-side STT
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-IN'; // Indic English

      recognition.onresult = (event) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setTranscript(currentTranscript);

        // If speech recognition finalized the sentence, send to RAG pipeline
        if (event.results[0].isFinal) {
          sendQueryToBackend(currentTranscript);
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error !== 'no-speech') {
          setError(`Speech error: ${event.error}`);
        }
        setIsRecording(false);
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current = recognition;
    } else {
      setError('Browser speech recognition not supported. Please use Google Chrome or Edge.');
    }

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (recognitionRef.current) recognitionRef.current.abort();
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
          return;
        }

        if (data.answer) {
          setResponse(data.answer);
          setLatency(data.latency_ms);
          setGuardrailTriggered(data.guardrail_triggered || false);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
      setError('WebSocket connection error. Is the FastAPI backend running?');
      setIsConnected(false);
    };

    ws.onclose = () => {
      setIsConnected(false);
      setTimeout(connectWebSocket, 3000);
    };

    wsRef.current = ws;
  };

  const sendQueryToBackend = (queryText) => {
    if (!queryText.trim()) return;

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ query: queryText }));
    } else {
      setError('WebSocket is not connected.');
    }
  };

  const startRecording = () => {
    setError(null);
    setResponse('');
    setTranscript('');
    setLatency(null);
    setGuardrailTriggered(false);

    if (recognitionRef.current) {
      try {
        recognitionRef.current.start();
        setIsRecording(true);
      } catch (err) {
        console.error('Recognition start error:', err);
      }
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsRecording(false);
  };

  return (
    <div className="w-full max-w-xl p-6 bg-slate-900 text-white rounded-2xl shadow-2xl border border-slate-800 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold tracking-wide">Voice-RAG Sub-200ms Interface</h2>
        <span
          className={`px-3 py-1 text-xs rounded-full font-medium ${
            isConnected ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
          }`}
        >
          {isConnected ? 'WS Connected' : 'Disconnected'}
        </span>
      </div>

      {error && (
        <div className="p-3 text-sm bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-lg">
          {error}
        </div>
      )}

      {/* Spoken Transcript Preview */}
      {transcript && (
        <div className="p-3 bg-slate-800/60 rounded-lg border border-slate-700/50">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold block mb-1">
            You Asked:
          </span>
          <p className="text-slate-200 italic">"{transcript}"</p>
        </div>
      )}

      {/* Voice Control Button */}
      <div className="flex justify-center py-2">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected}
          className={`px-8 py-4 rounded-full text-lg font-semibold transition-all transform active:scale-95 shadow-lg ${
            isRecording
              ? 'bg-rose-600 hover:bg-rose-700 animate-pulse text-white shadow-rose-600/30'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30 disabled:bg-slate-700'
          }`}
        >
          {isRecording ? '⏹ Stop & Analyze' : '🎙 Speak Question'}
        </button>
      </div>

      {/* Latency & Metrics */}
      {latency !== null && (
        <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg text-sm border border-slate-700">
          <span className="text-slate-400 font-medium">End-to-End Latency:</span>
          <span className={`font-mono font-bold text-base ${latency <= 200 ? 'text-emerald-400' : 'text-amber-400'}`}>
            {latency} ms
          </span>
        </div>
      )}

      {/* Guardrail Rejection Banner */}
      {guardrailTriggered && (
        <div className="p-3 text-sm bg-amber-500/10 border border-amber-500/30 text-amber-400 rounded-lg">
          ⚠️ <strong>Guardrail Triggered:</strong> Query similarity distance score fell below threshold (&lt; 0.60).
        </div>
      )}

      {/* Synthesized Answer */}
      {response && (
        <div className="p-4 bg-slate-800/90 rounded-xl border border-slate-700 space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
            Retrieved Answer
          </h3>
          <p className="text-slate-100 text-base leading-relaxed">{response}</p>
        </div>
      )}
    </div>
  );
}