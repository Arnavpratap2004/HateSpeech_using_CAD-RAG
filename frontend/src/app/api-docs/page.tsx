"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Card, Badge } from "@/components/ui";

export default function ApiDocsPage() {
    const endpoints = [
        {
            method: "POST",
            path: "/analyze",
            description: "Analyze text for hate speech using CAD-RAG",
            requestBody: {
                text: "string (required)",
                enable_rag: "boolean (optional, default: true)",
                threshold: "number (optional, default: 0.5)",
            },
            response: {
                prediction: "HATEFUL | NOT_HATEFUL",
                confidence: "number (0-1)",
                category: "string",
                labels: "string[]",
                probabilities: "Record<string, number>",
                rag_context: "string | null",
                llm_rationale: "string | null",
                entities: "string[]",
                neologisms: "string[]",
                indicator_matches: "Record<string, string[]>",
            },
        },
        {
            method: "GET",
            path: "/health",
            description: "Check API health and component status",
            requestBody: null,
            response: {
                status: "string",
                model_loaded: "boolean",
                rag_available: "boolean",
            },
        },
    ];

    return (
        <main className="min-h-screen bg-[#0B0F19]">
            {/* Header */}
            <header className="border-b border-gray-800">
                <div className="container mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-700 rounded-lg flex items-center justify-center">
                            <span className="text-white font-bold text-sm">C</span>
                        </div>
                        <span className="text-white font-semibold">CAD-RAG</span>
                    </Link>
                    <nav className="flex items-center gap-6">
                        <Link href="/analyze" className="text-gray-400 hover:text-white text-sm">Analyze</Link>
                        <Link href="/compare" className="text-gray-400 hover:text-white text-sm">Compare</Link>
                        <Link href="/dataset" className="text-gray-400 hover:text-white text-sm">Dataset</Link>
                        <Link href="/api-docs" className="text-blue-400 text-sm font-medium">API</Link>
                    </nav>
                </div>
            </header>

            <div className="container mx-auto px-6 py-12">
                {/* Hero */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mb-12"
                >
                    <h1 className="text-4xl font-bold text-white mb-4">API Documentation</h1>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Integrate CAD-RAG hate speech detection into your applications.
                    </p>
                    <div className="mt-6 flex items-center justify-center gap-4">
                        <Badge type="info">v1.0.0</Badge>
                        <span className="text-gray-500">Base URL: http://localhost:8000</span>
                    </div>
                </motion.div>

                {/* Quick Start */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="mb-12"
                >
                    <Card variant="gradient-border">
                        <h2 className="text-xl font-semibold text-white mb-4">Quick Start</h2>
                        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                            <pre className="text-sm text-gray-300">
                                <code>{`curl -X POST http://localhost:8000/analyze \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Your text to analyze here"}'`}</code>
                            </pre>
                        </div>
                    </Card>
                </motion.div>

                {/* Endpoints */}
                <div className="space-y-8">
                    {endpoints.map((endpoint, i) => (
                        <motion.div
                            key={endpoint.path}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 + i * 0.1 }}
                        >
                            <Card variant="glass">
                                {/* Endpoint Header */}
                                <div className="flex items-center gap-4 mb-6">
                                    <span className={`px-3 py-1 rounded-lg font-mono text-sm font-bold ${endpoint.method === "POST"
                                            ? "bg-green-500/20 text-green-400"
                                            : "bg-blue-500/20 text-blue-400"
                                        }`}>
                                        {endpoint.method}
                                    </span>
                                    <code className="text-white font-mono text-lg">{endpoint.path}</code>
                                </div>

                                <p className="text-gray-400 mb-6">{endpoint.description}</p>

                                <div className="grid md:grid-cols-2 gap-6">
                                    {/* Request */}
                                    {endpoint.requestBody && (
                                        <div>
                                            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
                                                Request Body
                                            </h3>
                                            <div className="bg-gray-900 rounded-lg p-4">
                                                <pre className="text-sm text-gray-300 overflow-x-auto">
                                                    <code>{`{
${Object.entries(endpoint.requestBody)
                                                            .map(([key, value]) => `  "${key}": ${value}`)
                                                            .join(',\n')}
}`}</code>
                                                </pre>
                                            </div>
                                        </div>
                                    )}

                                    {/* Response */}
                                    <div>
                                        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">
                                            Response
                                        </h3>
                                        <div className="bg-gray-900 rounded-lg p-4">
                                            <pre className="text-sm text-gray-300 overflow-x-auto">
                                                <code>{`{
${Object.entries(endpoint.response)
                                                        .map(([key, value]) => `  "${key}": ${value}`)
                                                        .join(',\n')}
}`}</code>
                                            </pre>
                                        </div>
                                    </div>
                                </div>
                            </Card>
                        </motion.div>
                    ))}
                </div>

                {/* Example Response */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="mt-12"
                >
                    <Card variant="glass">
                        <h2 className="text-xl font-semibold text-white mb-4">Example Response</h2>
                        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                            <pre className="text-sm text-gray-300">
                                <code>{`{
  "sentence": "All muslims should be removed",
  "prediction": "HATEFUL",
  "confidence": 0.92,
  "category": "Religion-based hate",
  "labels": ["identity_hate", "toxic"],
  "probabilities": {
    "toxic": 0.72,
    "severe_toxic": 0.15,
    "obscene": 0.23,
    "threat": 0.08,
    "insult": 0.45,
    "identity_hate": 0.92
  },
  "rag_context": "Lexicon Result: Discriminatory language targeting religious groups",
  "llm_rationale": "This content may be classified as hate speech based on contextual evidence...",
  "entities": ["Muslims"],
  "neologisms": [],
  "indicator_matches": {
    "Religion-based hate": ["muslim"]
  }
}`}</code>
                            </pre>
                        </div>
                    </Card>
                </motion.div>

                {/* Rate Limits */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="mt-8"
                >
                    <Card variant="glass">
                        <h2 className="text-xl font-semibold text-white mb-4">Rate Limits & Notes</h2>
                        <div className="grid md:grid-cols-3 gap-6">
                            <div>
                                <h3 className="text-sm font-medium text-gray-400 mb-2">Rate Limit</h3>
                                <p className="text-white">100 requests/minute</p>
                            </div>
                            <div>
                                <h3 className="text-sm font-medium text-gray-400 mb-2">Max Text Length</h3>
                                <p className="text-white">5,000 characters</p>
                            </div>
                            <div>
                                <h3 className="text-sm font-medium text-gray-400 mb-2">Response Time</h3>
                                <p className="text-white">~500ms (with RAG)</p>
                            </div>
                        </div>
                    </Card>
                </motion.div>
            </div>
        </main>
    );
}
