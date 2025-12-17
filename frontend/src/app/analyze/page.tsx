"use client";

import { useState, useEffect, Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { Button, Card, Badge, ConfidenceBar } from "@/components/ui";
import { ExplanationPanel, ChainOfDecision, ContextVisualizer } from "@/components";
import { api, AnalysisResponse } from "@/lib/api";

function AnalyzeContent() {
    const searchParams = useSearchParams();
    const initialText = searchParams.get("text") || "";

    const [inputText, setInputText] = useState(initialText);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState<AnalysisResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [enableRag, setEnableRag] = useState(true);
    const [analysisStep, setAnalysisStep] = useState(-1);
    const [activeTab, setActiveTab] = useState<"results" | "explanation" | "context">("results");

    useEffect(() => {
        if (initialText) {
            handleAnalyze();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleAnalyze = async () => {
        if (!inputText.trim()) return;

        setIsAnalyzing(true);
        setError(null);
        setResult(null);
        setAnalysisStep(0);

        // Simulate step progression for demo
        const stepTimers = [
            setTimeout(() => setAnalysisStep(1), 300),
            setTimeout(() => setAnalysisStep(2), 600),
            setTimeout(() => setAnalysisStep(3), 900),
            setTimeout(() => setAnalysisStep(4), 1200),
        ];

        try {
            const response = await api.analyze({
                text: inputText,
                enable_rag: enableRag,
            });
            setResult(response);
            setAnalysisStep(5);
        } catch {
            // Demo fallback
            await new Promise(resolve => setTimeout(resolve, 1500));
            const isHateful = inputText.toLowerCase().includes("hate") ||
                inputText.toLowerCase().includes("muslim") ||
                inputText.toLowerCase().includes("kill") ||
                inputText.toLowerCase().includes("remove");

            setResult({
                sentence: inputText,
                prediction: isHateful ? "HATEFUL" : "NOT_HATEFUL",
                confidence: isHateful ? 0.87 : 0.23,
                category: isHateful ? "Religion-based hate" : "General Hate",
                labels: isHateful ? ["identity_hate", "toxic"] : [],
                probabilities: {
                    toxic: isHateful ? 0.72 : 0.15,
                    severe_toxic: isHateful ? 0.15 : 0.02,
                    obscene: isHateful ? 0.23 : 0.08,
                    threat: isHateful ? 0.08 : 0.01,
                    insult: isHateful ? 0.45 : 0.12,
                    identity_hate: isHateful ? 0.87 : 0.05,
                },
                rag_context: isHateful
                    ? "Lexicon Result for 'religious group': Discriminatory language targeting religious communities | Contextual Result for 'Muslims': Historical pattern of exclusionary rhetoric | Counter-Evidence Result: No reclaimed usage found"
                    : "No specific entities or neologisms found to query.",
                llm_rationale: isHateful
                    ? "This content may be classified as hate speech based on contextual evidence. The sentence contains language that targets a religious group and expresses exclusionary sentiment, which aligns with patterns identified in our hate speech lexicon. The use of removal/deportation language combined with group targeting indicates identity-based discrimination."
                    : "This content appears to be neutral. No hate indicators or discriminatory patterns were detected in the analysis.",
                entities: isHateful ? ["Muslims", "Religious Group"] : [],
                neologisms: [],
                indicator_matches: isHateful ? {
                    "Religion-based hate": ["muslim"],
                } : {},
                // New CAD-RAG Final Decision fields
                final_label: isHateful ? "Hateful" : "Non-Hateful",
                override_pre_analysis: false,
                final_justification: isHateful
                    ? "The sentence explicitly calls for the removal of a religious group (Muslims) from the country, which constitutes hate speech targeting a protected group based on their religious identity. The retrieved context confirms this matches historical patterns of exclusionary and discriminatory rhetoric. No contextual evidence was found to neutralize or explain the usage in a non-hateful manner."
                    : "The content does not contain any hateful language or discriminatory intent. No contextual evidence suggests harmful targeting of any protected group.",
                pre_label: isHateful ? "HATEFUL" : "NOT_HATEFUL",
                pre_confidence: isHateful ? 0.87 : 0.23,
            });
            setAnalysisStep(5);
            setError("Demo mode: Backend not connected");
        } finally {
            setIsAnalyzing(false);
            stepTimers.forEach(timer => clearTimeout(timer));
        }
    };

    const getPredictionBadgeType = (prediction: string) => {
        return prediction === "HATEFUL" ? "hate" : "neutral";
    };

    return (
        <main className="min-h-screen bg-[#0B0F19]">
            {/* Header */}
            <header className="border-b border-gray-800 sticky top-0 bg-[#0B0F19]/80 backdrop-blur-lg z-50">
                <div className="container mx-auto px-6 py-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-700 rounded-lg flex items-center justify-center">
                            <span className="text-white font-bold text-sm">C</span>
                        </div>
                        <span className="text-white font-semibold">CAD-RAG</span>
                    </Link>
                    <nav className="flex items-center gap-6">
                        <Link href="/analyze" className="text-blue-400 text-sm font-medium">Analyze</Link>
                        <Link href="#" className="text-gray-400 hover:text-white text-sm">Compare</Link>
                        <Link href="#" className="text-gray-400 hover:text-white text-sm">Dataset</Link>
                        <Link href="#" className="text-gray-400 hover:text-white text-sm">API</Link>
                    </nav>
                </div>
            </header>

            {/* Main Content */}
            <div className="container mx-auto px-6 py-8">
                {/* Chain of Decision - Always visible when analyzing or has result */}
                {(isAnalyzing || result) && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-8"
                    >
                        <Card variant="glass">
                            <ChainOfDecision
                                currentStep={analysisStep}
                                isAnalyzing={isAnalyzing}
                                result={result ? {
                                    prediction: result.prediction,
                                    category: result.category,
                                    ragAvailable: !!result.rag_context && !result.rag_context.includes("No specific"),
                                    llmAvailable: !!result.llm_rationale && !result.llm_rationale.includes("not available"),
                                } : undefined}
                            />
                        </Card>
                    </motion.div>
                )}

                <div className="grid lg:grid-cols-5 gap-8">
                    {/* Left Panel: Input (2 cols) */}
                    <div className="lg:col-span-2">
                        <Card variant="glass" className="sticky top-24">
                            <h2 className="text-xl font-semibold text-white mb-6">Analyze Content</h2>

                            <div className="space-y-6">
                                <div>
                                    <label className="block text-sm text-gray-400 mb-2">Input Text</label>
                                    <textarea
                                        value={inputText}
                                        onChange={(e) => setInputText(e.target.value)}
                                        placeholder="Enter text to analyze for hate speech..."
                                        className="w-full h-40 bg-gray-900/50 border border-gray-700 rounded-xl p-4 text-white placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500 transition-colors"
                                    />
                                </div>

                                {/* Options */}
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-gray-400">Enable Context Retrieval (RAG)</span>
                                        <button
                                            onClick={() => setEnableRag(!enableRag)}
                                            className={`w-12 h-6 rounded-full transition-colors ${enableRag ? "bg-blue-600" : "bg-gray-700"
                                                } relative`}
                                        >
                                            <motion.div
                                                className="w-5 h-5 bg-white rounded-full absolute top-0.5"
                                                animate={{ left: enableRag ? "26px" : "2px" }}
                                                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                            />
                                        </button>
                                    </div>
                                </div>

                                <Button
                                    onClick={handleAnalyze}
                                    isLoading={isAnalyzing}
                                    className="w-full"
                                    size="lg"
                                    disabled={!inputText.trim()}
                                >
                                    {isAnalyzing ? "Analyzing..." : "Run CAD-RAG Analysis"}
                                </Button>

                                {error && (
                                    <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg text-yellow-400 text-sm">
                                        ⚠️ {error}
                                    </div>
                                )}
                            </div>
                        </Card>
                    </div>

                    {/* Right Panel: Results (3 cols) */}
                    <div className="lg:col-span-3 space-y-6">
                        <AnimatePresence mode="wait">
                            {isAnalyzing ? (
                                <motion.div
                                    key="loading"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="flex flex-col items-center justify-center h-80"
                                >
                                    <motion.div
                                        className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full"
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                    />
                                    <p className="text-gray-400 mt-4">
                                        {analysisStep === 0 && "Extracting entities..."}
                                        {analysisStep === 1 && "Retrieving context..."}
                                        {analysisStep === 2 && "Running ML model..."}
                                        {analysisStep === 3 && "Generating rationale..."}
                                        {analysisStep === 4 && "Finalizing decision..."}
                                    </p>
                                </motion.div>
                            ) : result ? (
                                <motion.div
                                    key="results"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="space-y-6"
                                >
                                    {/* Tab Navigation */}
                                    <div className="flex gap-2 p-1 bg-gray-900/50 rounded-xl">
                                        {[
                                            { id: "results", label: "Results", icon: "📊" },
                                            { id: "explanation", label: "Explanation", icon: "🔍" },
                                            { id: "context", label: "Context", icon: "📚" },
                                        ].map((tab) => (
                                            <button
                                                key={tab.id}
                                                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                                                className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2 ${activeTab === tab.id
                                                    ? "bg-blue-600 text-white"
                                                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                                                    }`}
                                            >
                                                <span>{tab.icon}</span>
                                                {tab.label}
                                            </button>
                                        ))}
                                    </div>

                                    {/* Tab Content */}
                                    <AnimatePresence mode="wait">
                                        {activeTab === "results" && (
                                            <motion.div
                                                key="results-tab"
                                                initial={{ opacity: 0, x: 20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                exit={{ opacity: 0, x: -20 }}
                                                className="space-y-6"
                                            >
                                                {/* Final Decision Card - Context-Aware Result */}
                                                {result.final_label && (
                                                    <Card variant="gradient-border" className="relative overflow-hidden">
                                                        <div className="flex items-center gap-3 mb-4">
                                                            <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl flex items-center justify-center">
                                                                <span className="text-white text-lg">⚖️</span>
                                                            </div>
                                                            <div>
                                                                <h3 className="text-lg font-semibold text-white">Final Decision</h3>
                                                                <p className="text-xs text-gray-400">Context-aware classification</p>
                                                            </div>
                                                        </div>

                                                        <div className="flex items-center justify-between">
                                                            <div className="flex items-center gap-3">
                                                                <span className={`px-4 py-2 rounded-full font-bold text-sm ${result.final_label === "Hateful"
                                                                    ? "bg-red-500/20 text-red-400 border border-red-500/30"
                                                                    : "bg-green-500/20 text-green-400 border border-green-500/30"
                                                                    }`}>
                                                                    {result.final_label === "Hateful" ? "⚠️ Hateful" : "✓ Non-Hateful"}
                                                                </span>
                                                                {result.override_pre_analysis && (
                                                                    <span className="px-3 py-1 bg-amber-500/20 text-amber-400 text-xs font-medium rounded-full border border-amber-500/30">
                                                                        Pre-analysis overridden
                                                                    </span>
                                                                )}
                                                            </div>
                                                            <button
                                                                type="button"
                                                                onClick={() => setActiveTab("explanation")}
                                                                className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1 cursor-pointer relative z-10"
                                                            >
                                                                View justification →
                                                            </button>
                                                        </div>
                                                    </Card>
                                                )}

                                                {/* Prediction Card (Pre-retrieval) */}
                                                <Card variant="glass">
                                                    <div className="flex items-center justify-between mb-4">
                                                        <h3 className="text-lg font-semibold text-white">Pre-Analysis Prediction</h3>
                                                        <Badge type={getPredictionBadgeType(result.prediction)}>
                                                            {result.prediction === "HATEFUL" ? "⚠️ Hate Speech" : "✓ Neutral"}
                                                        </Badge>
                                                    </div>
                                                    <ConfidenceBar
                                                        value={result.confidence * 100}
                                                        label="Confidence"
                                                        colorScheme={result.prediction === "HATEFUL" ? "danger" : "success"}
                                                    />
                                                </Card>

                                                {/* Category Card */}
                                                <Card variant="glass">
                                                    <h3 className="text-lg font-semibold text-white mb-4">Category Detection</h3>
                                                    <div className="flex flex-wrap gap-2">
                                                        <Badge type="info">{result.category}</Badge>
                                                        {result.labels?.map((label) => (
                                                            <Badge key={label} type="offensive" size="sm">
                                                                {label.replace(/_/g, " ")}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </Card>

                                                {/* Probabilities */}
                                                {result.probabilities && (
                                                    <Card variant="glass">
                                                        <h3 className="text-lg font-semibold text-white mb-4">Label Probabilities</h3>
                                                        <div className="space-y-3">
                                                            {Object.entries(result.probabilities)
                                                                .sort(([, a], [, b]) => b - a)
                                                                .map(([label, prob]) => (
                                                                    <ConfidenceBar
                                                                        key={label}
                                                                        value={prob * 100}
                                                                        label={label.replace(/_/g, " ")}
                                                                    />
                                                                ))}
                                                        </div>
                                                    </Card>
                                                )}
                                            </motion.div>
                                        )}

                                        {activeTab === "explanation" && (
                                            <motion.div
                                                key="explanation-tab"
                                                initial={{ opacity: 0, x: 20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                exit={{ opacity: 0, x: -20 }}
                                            >
                                                <ExplanationPanel
                                                    ragContext={result.rag_context}
                                                    llmRationale={result.llm_rationale}
                                                    indicatorMatches={result.indicator_matches}
                                                    entities={result.entities}
                                                    finalLabel={result.final_label}
                                                    overridePreAnalysis={result.override_pre_analysis}
                                                    finalJustification={result.final_justification}
                                                    preLabel={result.pre_label}
                                                />
                                            </motion.div>
                                        )}

                                        {activeTab === "context" && (
                                            <motion.div
                                                key="context-tab"
                                                initial={{ opacity: 0, x: 20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                exit={{ opacity: 0, x: -20 }}
                                            >
                                                <ContextVisualizer
                                                    entities={result.entities}
                                                    neologisms={result.neologisms}
                                                    ragContext={result.rag_context}
                                                />
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </motion.div>
                            ) : (
                                <motion.div
                                    key="empty"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    className="flex flex-col items-center justify-center h-80 text-center"
                                >
                                    <div className="text-6xl mb-4">🔍</div>
                                    <h3 className="text-xl font-semibold text-white mb-2">Ready to Analyze</h3>
                                    <p className="text-gray-400 max-w-sm">
                                        Enter text in the input panel and click &ldquo;Run CAD-RAG Analysis&rdquo; to get started.
                                    </p>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </main>
    );
}

export default function AnalyzePage() {
    return (
        <Suspense fallback={
            <div className="min-h-screen bg-[#0B0F19] flex items-center justify-center">
                <div className="w-8 h-8 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
            </div>
        }>
            <AnalyzeContent />
        </Suspense>
    );
}
