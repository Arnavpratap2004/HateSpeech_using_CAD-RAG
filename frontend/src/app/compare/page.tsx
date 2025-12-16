"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { Button, Card, Badge, ConfidenceBar } from "@/components/ui";

interface ComparisonResult {
    sentence: string;
    traditional: {
        prediction: string;
        confidence: number;
        labels: string[];
        hasContext: boolean;
        hasExplanation: boolean;
    };
    cadRag: {
        prediction: string;
        confidence: number;
        labels: string[];
        hasContext: boolean;
        hasExplanation: boolean;
        category: string;
        rationale: string;
    };
}

export default function ComparePage() {
    const [inputText, setInputText] = useState("");
    const [isComparing, setIsComparing] = useState(false);
    const [result, setResult] = useState<ComparisonResult | null>(null);

    const handleCompare = async () => {
        if (!inputText.trim()) return;

        setIsComparing(true);

        // Simulate comparison
        await new Promise(resolve => setTimeout(resolve, 2000));

        const isHateful = inputText.toLowerCase().includes("hate") ||
            inputText.toLowerCase().includes("muslim") ||
            inputText.toLowerCase().includes("kill");

        setResult({
            sentence: inputText,
            traditional: {
                prediction: isHateful ? "HATEFUL" : "NOT_HATEFUL",
                confidence: isHateful ? 0.72 : 0.45,
                labels: isHateful ? ["toxic"] : [],
                hasContext: false,
                hasExplanation: false,
            },
            cadRag: {
                prediction: isHateful ? "HATEFUL" : "NOT_HATEFUL",
                confidence: isHateful ? 0.92 : 0.15,
                labels: isHateful ? ["identity_hate", "toxic"] : [],
                hasContext: true,
                hasExplanation: true,
                category: isHateful ? "Religion-based hate" : "General",
                rationale: isHateful
                    ? "Detected discriminatory language targeting religious groups with exclusionary intent."
                    : "No hate indicators detected. Content appears neutral.",
            },
        });

        setIsComparing(false);
    };

    const comparisonFeatures = [
        { feature: "Context Awareness", traditional: false, cadRag: true, description: "Uses RAG to retrieve relevant context" },
        { feature: "Explainability", traditional: false, cadRag: true, description: "Provides detailed rationale for decisions" },
        { feature: "Bias Reduction", traditional: "partial", cadRag: true, description: "Multi-source validation reduces bias" },
        { feature: "Evolving Terms", traditional: false, cadRag: true, description: "Detects neologisms and coded language" },
        { feature: "Target Detection", traditional: false, cadRag: true, description: "Identifies specific targeted groups" },
        { feature: "Severity Assessment", traditional: false, cadRag: true, description: "Evaluates severity level of content" },
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
                        <Link href="/compare" className="text-blue-400 text-sm font-medium">Compare</Link>
                        <Link href="/dataset" className="text-gray-400 hover:text-white text-sm">Dataset</Link>
                        <Link href="/api-docs" className="text-gray-400 hover:text-white text-sm">API</Link>
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
                    <h1 className="text-4xl font-bold text-white mb-4">
                        CAD-RAG vs Traditional Models
                    </h1>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        See how Context-Aware Retrieval Augmented Generation outperforms traditional
                        hate speech detection approaches.
                    </p>
                </motion.div>

                {/* Feature Comparison Table */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="mb-12"
                >
                    <Card variant="glass">
                        <h2 className="text-xl font-semibold text-white mb-6">Feature Comparison</h2>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-gray-800">
                                        <th className="text-left py-3 px-4 text-gray-400 font-medium">Feature</th>
                                        <th className="text-center py-3 px-4 text-gray-400 font-medium">Traditional ML</th>
                                        <th className="text-center py-3 px-4 text-gray-400 font-medium">CAD-RAG</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {comparisonFeatures.map((item, i) => (
                                        <motion.tr
                                            key={item.feature}
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: 0.2 + i * 0.05 }}
                                            className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
                                        >
                                            <td className="py-4 px-4">
                                                <div>
                                                    <div className="text-white font-medium">{item.feature}</div>
                                                    <div className="text-xs text-gray-500">{item.description}</div>
                                                </div>
                                            </td>
                                            <td className="py-4 px-4 text-center">
                                                {item.traditional === true ? (
                                                    <span className="text-green-400 text-xl">✓</span>
                                                ) : item.traditional === "partial" ? (
                                                    <span className="text-yellow-400 text-xl">⚠️</span>
                                                ) : (
                                                    <span className="text-red-400 text-xl">✗</span>
                                                )}
                                            </td>
                                            <td className="py-4 px-4 text-center">
                                                <span className="text-green-400 text-xl">✓</span>
                                            </td>
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </Card>
                </motion.div>

                {/* Live Comparison Demo */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <Card variant="gradient-border">
                        <h2 className="text-xl font-semibold text-white mb-6">Live Comparison Demo</h2>

                        {/* Input */}
                        <div className="mb-6">
                            <textarea
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                placeholder="Enter text to compare both approaches..."
                                className="w-full h-24 bg-gray-900/50 border border-gray-700 rounded-xl p-4 text-white placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500"
                            />
                            <Button
                                onClick={handleCompare}
                                isLoading={isComparing}
                                className="mt-4"
                                disabled={!inputText.trim()}
                            >
                                Compare Approaches
                            </Button>
                        </div>

                        {/* Results */}
                        {result && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="grid md:grid-cols-2 gap-6"
                            >
                                {/* Traditional */}
                                <Card variant="glass" className="border-gray-600">
                                    <div className="flex items-center justify-between mb-4">
                                        <h3 className="text-lg font-semibold text-white">Traditional ML</h3>
                                        <Badge type={result.traditional.prediction === "HATEFUL" ? "hate" : "neutral"}>
                                            {result.traditional.prediction}
                                        </Badge>
                                    </div>

                                    <ConfidenceBar
                                        value={result.traditional.confidence * 100}
                                        label="Confidence"
                                    />

                                    <div className="mt-4 space-y-2">
                                        <div className="flex items-center gap-2 text-sm">
                                            <span className="text-red-400">✗</span>
                                            <span className="text-gray-400">No context retrieval</span>
                                        </div>
                                        <div className="flex items-center gap-2 text-sm">
                                            <span className="text-red-400">✗</span>
                                            <span className="text-gray-400">No explanation provided</span>
                                        </div>
                                        <div className="flex items-center gap-2 text-sm">
                                            <span className="text-red-400">✗</span>
                                            <span className="text-gray-400">No target detection</span>
                                        </div>
                                    </div>
                                </Card>

                                {/* CAD-RAG */}
                                <Card variant="glass" className="border-blue-500/50 shadow-lg shadow-blue-500/10">
                                    <div className="flex items-center justify-between mb-4">
                                        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                            CAD-RAG
                                            <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 text-xs rounded">Enhanced</span>
                                        </h3>
                                        <Badge type={result.cadRag.prediction === "HATEFUL" ? "hate" : "neutral"}>
                                            {result.cadRag.prediction}
                                        </Badge>
                                    </div>

                                    <ConfidenceBar
                                        value={result.cadRag.confidence * 100}
                                        label="Confidence"
                                        colorScheme={result.cadRag.prediction === "HATEFUL" ? "danger" : "success"}
                                    />

                                    <div className="mt-4 space-y-2">
                                        <div className="flex items-center gap-2 text-sm">
                                            <span className="text-green-400">✓</span>
                                            <span className="text-gray-400">Context retrieved from knowledge base</span>
                                        </div>
                                        <div className="flex items-center gap-2 text-sm">
                                            <span className="text-green-400">✓</span>
                                            <span className="text-gray-400">Category: {result.cadRag.category}</span>
                                        </div>
                                        <div className="flex items-center gap-2 text-sm">
                                            <span className="text-green-400">✓</span>
                                            <span className="text-gray-400">Labels: {result.cadRag.labels.join(", ") || "None"}</span>
                                        </div>
                                    </div>

                                    <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                                        <p className="text-sm text-gray-300 italic">&ldquo;{result.cadRag.rationale}&rdquo;</p>
                                    </div>
                                </Card>
                            </motion.div>
                        )}
                    </Card>
                </motion.div>
            </div>
        </main>
    );
}
