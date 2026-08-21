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
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);

  // Initialize WebSocket connection on component mount
  useEffect(() => {
    connectWebSocket();

    return () => {
      stopRecording();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
      console.log('Connected to Voice-RAG WebSocket server');
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
      console.log('WebSocket connection closed. Reconnecting in 3s...');
      setTimeout(connectWebSocket, 3000);
    };

    wsRef.current = ws;
  };

  // Convert Float32 audio samples from Web Audio API to 16-bit PCM Base64 string
  const convertFloat32ToPCM16Base64 = (buffer) => {
    const l = buffer.length;
    const buf = new Int16Array(l);
    for (let i = 0; i < l; i++) {
      const s = Math.max(-1, Math.min(1, buffer[i]));
      buf[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    
    let binary = '';
    const bytes = new Uint8Array(buf.buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  };

  const startRecording = async () => {
    setError(null);
    setResponse('');
    setLatency(null);
    setGuardrailTriggered(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000, // Explicit 16kHz sample rate matching STT requirements
      });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      // Processing 4096 frames (~256ms buffer) per chunk
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const base64Audio = convertFloat32ToPCM16Base64(inputData);

        // Stream audio chunk payload over WebSocket
        wsRef.current.send(
          JSON.stringify({
            event: 'media',
            media: { payload: base64Audio },
          })
        );
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Microphone access denied or unreadable.');
    }
  };

  const stopRecording = () => {
    if (processorRef.current && sourceRef?.current) {
      processorRef.current.disconnect();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    setIsRecording(false);

    // Send end-of-speech signal to flush STT buffer and finalize pipeline
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ event: 'stop' }));
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6 bg-slate-900 text-white rounded-xl shadow-lg border border-slate-800 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Voice-RAG Sub-200ms Interface</h2>
        <span
          className={`px-3 py-1 text-xs rounded-full font-medium ${
            isConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
          }`}
        >
          {isConnected ? 'WS Connected' : 'Disconnected'}
        </span>
      </div>

      {error && <div className="p-3 text-sm bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-md">{error}</div>}

      <div className="flex justify-center py-4">
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected}
          className={`relative px-8 py-4 rounded-full text-lg font-semibold transition-all ${
            isRecording
              ? 'bg-rose-600 hover:bg-rose-700 animate-pulse text-white'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white disabled:bg-slate-700'
          }`}
        >
          {isRecording ? '⏹ Stop Recording' : '🎙 Speak Question'}
        </button>
      </div>

      {/* Output & Latency Display */}
      <div className="space-y-4">
        {latency !== null && (
          <div className="flex items-center justify-between p-3 bg-slate-800/80 rounded-lg text-sm">
            <span>Latency Metric:</span>
            <span className={`font-mono font-bold ${latency <= 200 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {latency} ms
            </span>
          </div>
        )}

        {guardrailTriggered && (
          <div className="p-3 text-sm bg-amber-500/10 border border-amber-500/30 text-amber-400 rounded-md">
            ⚠️ Guardrail Triggered: Query out-of-context or failed grounding similarity threshold.
          </div>
        )}

        {response && (
          <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 space-y-2">
            <h3 className="text-xs font-semibold uppercase text-slate-400">Retrieved Answer</h3>
            <p className="text-slate-100 text-base leading-relaxed">{response}</p>
          </div>
        )}
      </div>
    </div>
  );
}