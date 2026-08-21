import React from 'react';
import AudioRecorder from './components/AudioRecorder';

export default function App() {
  return (
    <main className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          HH Goa 2026 Voice RAG
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Sub-200ms Voice-Enabled Indic RAG with Guardrails
        </p>
      </header>
      
      <AudioRecorder />
    </main>
  );
}