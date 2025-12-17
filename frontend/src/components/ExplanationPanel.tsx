"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui";

interface RAGEvidence {
    source: string;
    context: string;
    relevanceScore: number;
    matchedPhrases?: string[];
}

interface ExplanationPanelProps {
    ragContext?: string;
    llmRationale?: string;
    indicatorMatches?: Record<string, string[]>;
    entities?: string[];
    // New CAD-RAG Final Decision props
    finalLabel?: string;
    overridePreAnalysis?: boolean;
    finalJustification?: string;
    preLabel?: string;
}

export default function ExplanationPanel({
    ragContext,
    llmRationale,
    indicatorMatches,
    entities,
    finalLabel,
    overridePreAnalysis,
    finalJustification,
    preLabel,
}: ExplanationPanelProps) {
    // Parse RAG context into evidence cards
    const parseEvidenceCards = (context: string): RAGEvidence[] => {
        if (!context) return [];

        // Split by pipe delimiter used in engine.py
        const parts = context.split(" | ");
        return parts.map((part, i) => {
            const isLexicon = part.includes("Lexicon Result");
            const isContextual = part.includes("Contextual Result");
            const isCounter = part.includes("Counter-Evidence");

            let source = "Knowledge Base";
            if (isLexicon) source = "Slur Lexicon";
            else if (isContextual) source = "News Context";
            else if (isCounter) source = "Reclaimed Speech";

            // Extract relevance score (mock for now, would come from backend)
            const relevanceScore = 0.85 - (i * 0.1);

            return {
                source,
                context: part,
                relevanceScore: Math.max(0.5, relevanceScore),
            };
        }).filter(e => !e.context.includes("No specific entities"));
    };

    const evidenceCards = parseEvidenceCards(ragContext || "");

    const getFinalLabelColor = (label?: string) => {
        if (!label) return "gray";
        return label === "Hateful" ? "red" : "green";
    };

    return (
        <div className="space-y-6">
            {/* Context-Aware Final Decision Section */}
            {finalLabel && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-3"
                >
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl flex items-center justify-center">
                            <span className="text-white text-lg">⚖️</span>
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-white">Context-Aware Decision</h2>
                            <p className="text-sm text-gray-400">Final classification using retrieved evidence</p>
                        </div>
                    </div>

                    <Card variant="glass" className="relative overflow-hidden">
                        {/* Override Banner */}
                        {overridePreAnalysis && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                className="mb-4 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-center gap-3"
                            >
                                <span className="text-amber-400 text-xl">⚠️</span>
                                <div>
                                    <p className="text-amber-300 font-medium text-sm">Pre-Analysis Overridden</p>
                                    <p className="text-amber-400/70 text-xs">
                                        Context changed the classification from <span className="font-semibold">{preLabel}</span> to <span className="font-semibold">{finalLabel}</span>
                                    </p>
                                </div>
                            </motion.div>
                        )}

                        {/* Final Label Display */}
                        <div className="flex items-center justify-between mb-4">
                            <span className="text-gray-400 text-sm font-medium">Final Classification</span>
                            <span className={`px-4 py-2 rounded-full font-bold text-sm ${finalLabel === "Hateful"
                                    ? "bg-red-500/20 text-red-400 border border-red-500/30"
                                    : "bg-green-500/20 text-green-400 border border-green-500/30"
                                }`}>
                                {finalLabel === "Hateful" ? "⚠️ Hateful" : "✓ Non-Hateful"}
                            </span>
                        </div>

                        {/* Justification */}
                        {finalJustification && (
                            <div className="mt-4 pt-4 border-t border-gray-700/50">
                                <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Justification</p>
                                <p className="text-gray-300 text-sm leading-relaxed">
                                    {finalJustification}
                                </p>
                            </div>
                        )}

                        {/* Decorative gradient line */}
                        <div className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${finalLabel === "Hateful"
                                ? "from-transparent via-red-500/50 to-transparent"
                                : "from-transparent via-green-500/50 to-transparent"
                            }`} />
                    </Card>
                </motion.div>
            )}

            {/* Section Header */}
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                    <span className="text-white text-lg">🔍</span>
                </div>
                <div>
                    <h2 className="text-xl font-bold text-white">Why was this flagged?</h2>
                    <p className="text-sm text-gray-400">Evidence-based explanation from CAD-RAG</p>
                </div>
            </div>

            {/* RAG Evidence Cards */}
            {evidenceCards.length > 0 && (
                <div className="space-y-4">
                    <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
                        Retrieved Evidence
                    </h3>
                    {evidenceCards.map((evidence, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <Card variant="glass" className="relative overflow-hidden">
                                {/* Source Badge */}
                                <div className="flex items-center justify-between mb-3">
                                    <span className="px-3 py-1 bg-blue-500/20 text-blue-400 text-xs font-medium rounded-full">
                                        {evidence.source}
                                    </span>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-gray-500">Relevance</span>
                                        <div className="flex items-center gap-1">
                                            <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                                <motion.div
                                                    className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${evidence.relevanceScore * 100}%` }}
                                                    transition={{ delay: 0.3 + index * 0.1, duration: 0.5 }}
                                                />
                                            </div>
                                            <span className="text-xs text-gray-400">
                                                {(evidence.relevanceScore * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* Context Text */}
                                <p className="text-gray-300 text-sm leading-relaxed">
                                    {evidence.context}
                                </p>

                                {/* Matched Phrases Highlight */}
                                {evidence.matchedPhrases && evidence.matchedPhrases.length > 0 && (
                                    <div className="mt-3 flex flex-wrap gap-2">
                                        {evidence.matchedPhrases.map((phrase, i) => (
                                            <span
                                                key={i}
                                                className="px-2 py-0.5 bg-red-500/20 text-red-400 text-xs rounded"
                                            >
                                                {phrase}
                                            </span>
                                        ))}
                                    </div>
                                )}

                                {/* Decorative gradient line */}
                                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
                            </Card>
                        </motion.div>
                    ))}
                </div>
            )}

            {/* Indicator Matches */}
            {indicatorMatches && Object.keys(indicatorMatches).length > 0 && (
                <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
                        Matched Hate Indicators
                    </h3>
                    <Card variant="glass">
                        <div className="space-y-3">
                            {Object.entries(indicatorMatches).map(([category, terms]) => (
                                <div key={category}>
                                    <div className="flex items-center gap-2 mb-2">
                                        <span className="w-2 h-2 bg-red-500 rounded-full" />
                                        <span className="text-sm font-medium text-white">{category}</span>
                                    </div>
                                    <div className="flex flex-wrap gap-2 pl-4">
                                        {terms.map((term, i) => (
                                            <motion.span
                                                key={i}
                                                initial={{ opacity: 0, scale: 0.8 }}
                                                animate={{ opacity: 1, scale: 1 }}
                                                transition={{ delay: i * 0.05 }}
                                                className="px-2 py-1 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded-lg"
                                            >
                                                &ldquo;{term}&rdquo;
                                            </motion.span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>
            )}

            {/* Detected Entities */}
            {entities && entities.length > 0 && (
                <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
                        Detected Entities
                    </h3>
                    <Card variant="glass">
                        <div className="flex flex-wrap gap-2">
                            {entities.map((entity, i) => (
                                <motion.span
                                    key={entity}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.05 }}
                                    className="px-3 py-1.5 bg-purple-500/10 border border-purple-500/30 text-purple-300 text-sm rounded-lg flex items-center gap-2"
                                >
                                    <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
                                    {entity}
                                </motion.span>
                            ))}
                        </div>
                    </Card>
                </div>
            )}

            {/* LLM Rationale */}
            {llmRationale && !llmRationale.includes("not available") && (
                <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">
                        AI Analysis
                    </h3>
                    <Card variant="gradient-border" className="relative">
                        <div className="flex items-start gap-3">
                            <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-lg flex items-center justify-center flex-shrink-0">
                                <span className="text-white text-sm">🧠</span>
                            </div>
                            <div>
                                <p className="text-gray-300 text-sm leading-relaxed italic">
                                    &ldquo;{llmRationale}&rdquo;
                                </p>
                            </div>
                        </div>
                    </Card>
                </div>
            )}
        </div>
    );
}
