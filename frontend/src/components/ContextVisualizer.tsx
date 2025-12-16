"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui";

interface SimilarSentence {
    text: string;
    similarity: number;
    source: string;
    category?: string;
}

interface ContextVisualizerProps {
    entities?: string[];
    neologisms?: string[];
    ragContext?: string;
}

export default function ContextVisualizer({
    entities = [],
    neologisms = [],
    ragContext,
}: ContextVisualizerProps) {
    // Mock similar sentences (in production, would come from backend)
    const similarSentences: SimilarSentence[] = [
        {
            text: "Discriminatory statements targeting religious groups have been identified as harmful content.",
            similarity: 0.91,
            source: "Hatebase Lexicon",
            category: "Religion-based",
        },
        {
            text: "Language expressing exclusionary sentiment towards ethnic minorities.",
            similarity: 0.87,
            source: "Policy Database",
            category: "Ethnicity-based",
        },
        {
            text: "Content promoting removal or deportation based on identity characteristics.",
            similarity: 0.84,
            source: "Academic Corpus",
            category: "Nationality-based",
        },
    ];

    const getSimilarityColor = (score: number) => {
        if (score >= 0.9) return "from-red-500 to-red-400";
        if (score >= 0.8) return "from-orange-500 to-yellow-400";
        return "from-yellow-500 to-green-400";
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-teal-500 rounded-xl flex items-center justify-center">
                    <span className="text-white text-lg">📊</span>
                </div>
                <div>
                    <h2 className="text-xl font-bold text-white">Context Similarity</h2>
                    <p className="text-sm text-gray-400">Top-k similar patterns from knowledge base</p>
                </div>
            </div>

            {/* Similar Sentences */}
            <div className="space-y-4">
                {similarSentences.map((sentence, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.15 }}
                    >
                        <Card variant="glass" className="relative overflow-hidden">
                            {/* Similarity Indicator */}
                            <div className="absolute top-0 left-0 bottom-0 w-1">
                                <div
                                    className={`h-full bg-gradient-to-b ${getSimilarityColor(sentence.similarity)}`}
                                    style={{ height: `${sentence.similarity * 100}%` }}
                                />
                            </div>

                            <div className="pl-4">
                                {/* Header Row */}
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-2">
                                        <span className="px-2 py-0.5 bg-gray-800 text-gray-400 text-xs rounded">
                                            {sentence.source}
                                        </span>
                                        {sentence.category && (
                                            <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded">
                                                {sentence.category}
                                            </span>
                                        )}
                                    </div>

                                    {/* Similarity Score */}
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs text-gray-500">Similarity</span>
                                        <motion.div
                                            className={`px-3 py-1 rounded-full bg-gradient-to-r ${getSimilarityColor(sentence.similarity)} text-white text-sm font-bold`}
                                            initial={{ scale: 0 }}
                                            animate={{ scale: 1 }}
                                            transition={{ delay: 0.3 + index * 0.1, type: "spring" }}
                                        >
                                            {(sentence.similarity * 100).toFixed(0)}%
                                        </motion.div>
                                    </div>
                                </div>

                                {/* Sentence Text */}
                                <p className="text-gray-300 text-sm leading-relaxed">
                                    &ldquo;{sentence.text}&rdquo;
                                </p>
                            </div>
                        </Card>
                    </motion.div>
                ))}
            </div>

            {/* Extracted Terms Section */}
            {(entities.length > 0 || neologisms.length > 0) && (
                <div className="grid md:grid-cols-2 gap-4 mt-6">
                    {/* Entities */}
                    {entities.length > 0 && (
                        <Card variant="glass">
                            <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                                <span>👤</span> Detected Entities
                            </h4>
                            <div className="flex flex-wrap gap-2">
                                {entities.map((entity, i) => (
                                    <motion.span
                                        key={entity}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        transition={{ delay: i * 0.05 }}
                                        className="px-3 py-1 bg-blue-500/10 border border-blue-500/30 text-blue-300 text-sm rounded-lg"
                                    >
                                        {entity}
                                    </motion.span>
                                ))}
                            </div>
                        </Card>
                    )}

                    {/* Neologisms/Coded Terms */}
                    {neologisms.length > 0 && (
                        <Card variant="glass">
                            <h4 className="text-sm font-medium text-gray-400 mb-3 flex items-center gap-2">
                                <span>🔤</span> Coded Terms
                            </h4>
                            <div className="flex flex-wrap gap-2">
                                {neologisms.map((term, i) => (
                                    <motion.span
                                        key={term}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        transition={{ delay: i * 0.05 }}
                                        className="px-3 py-1 bg-orange-500/10 border border-orange-500/30 text-orange-300 text-sm rounded-lg"
                                    >
                                        {term}
                                    </motion.span>
                                ))}
                            </div>
                        </Card>
                    )}
                </div>
            )}

            {/* Knowledge Graph Placeholder */}
            <Card variant="glass" className="mt-6">
                <div className="flex items-center justify-between mb-4">
                    <h4 className="text-sm font-medium text-gray-400 flex items-center gap-2">
                        <span>🕸️</span> Knowledge Graph View
                    </h4>
                    <span className="px-2 py-1 bg-gray-800 text-gray-500 text-xs rounded">Coming Soon</span>
                </div>
                <div className="h-40 bg-gray-900/50 rounded-xl border border-gray-800 flex items-center justify-center">
                    <div className="text-center">
                        <div className="text-4xl mb-2 opacity-30">🔗</div>
                        <p className="text-gray-500 text-sm">Interactive knowledge graph visualization</p>
                    </div>
                </div>
            </Card>
        </div>
    );
}
