"use client";

import { motion } from "framer-motion";
import { useState } from "react";

interface Step {
    id: string;
    label: string;
    icon: string;
    description: string;
    status: "pending" | "active" | "complete";
}

interface ChainOfDecisionProps {
    currentStep?: number;
    isAnalyzing?: boolean;
    result?: {
        prediction: string;
        category: string;
        ragAvailable: boolean;
        llmAvailable: boolean;
    };
}

export default function ChainOfDecision({
    currentStep = -1,
    isAnalyzing = false,
    result,
}: ChainOfDecisionProps) {
    const [hoveredStep, setHoveredStep] = useState<string | null>(null);

    const steps: Step[] = [
        {
            id: "input",
            label: "Input",
            icon: "📝",
            description: "Text received for analysis",
            status: currentStep >= 0 ? "complete" : "pending",
        },
        {
            id: "preretrieval",
            label: "Pre-Retrieval",
            icon: "🔍",
            description: "Entity extraction & indicator detection",
            status: currentStep >= 1 ? "complete" : currentStep === 0 ? "active" : "pending",
        },
        {
            id: "context",
            label: "Context Retrieval",
            icon: "📚",
            description: "RAG queries knowledge bases",
            status: currentStep >= 2 ? "complete" : currentStep === 1 ? "active" : "pending",
        },
        {
            id: "ml",
            label: "ML Model",
            icon: "🤖",
            description: "Multi-label classification",
            status: currentStep >= 3 ? "complete" : currentStep === 2 ? "active" : "pending",
        },
        {
            id: "llm",
            label: "LLM Rationale",
            icon: "🧠",
            description: "Expert explanation generation",
            status: currentStep >= 4 ? "complete" : currentStep === 3 ? "active" : "pending",
        },
        {
            id: "decision",
            label: "Decision",
            icon: result?.prediction === "HATEFUL" ? "⚠️" : "✓",
            description: "Final classification with evidence",
            status: currentStep >= 5 ? "complete" : currentStep === 4 ? "active" : "pending",
        },
    ];

    const getStatusColor = (status: Step["status"]) => {
        switch (status) {
            case "complete":
                return "bg-green-500";
            case "active":
                return "bg-blue-500 animate-pulse";
            default:
                return "bg-gray-700";
        }
    };

    const getStatusBorder = (status: Step["status"]) => {
        switch (status) {
            case "complete":
                return "border-green-500/50";
            case "active":
                return "border-blue-500 shadow-lg shadow-blue-500/30";
            default:
                return "border-gray-700";
        }
    };

    return (
        <div className="w-full">
            {/* Header */}
            <div className="flex items-center gap-2 mb-6">
                <span className="text-lg">⛓️</span>
                <h3 className="text-lg font-semibold text-white">Chain of Decision</h3>
            </div>

            {/* Horizontal Flow */}
            <div className="relative">
                {/* Connection Line */}
                <div className="absolute top-8 left-0 right-0 h-0.5 bg-gray-800" />
                <motion.div
                    className="absolute top-8 left-0 h-0.5 bg-gradient-to-r from-blue-500 to-green-500"
                    initial={{ width: "0%" }}
                    animate={{ width: `${Math.max(0, (currentStep / 5) * 100)}%` }}
                    transition={{ duration: 0.5 }}
                />

                {/* Steps */}
                <div className="relative flex justify-between">
                    {steps.map((step, index) => (
                        <motion.div
                            key={step.id}
                            className="flex flex-col items-center"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            onMouseEnter={() => setHoveredStep(step.id)}
                            onMouseLeave={() => setHoveredStep(null)}
                        >
                            {/* Node */}
                            <motion.div
                                className={`w-16 h-16 rounded-2xl border-2 ${getStatusBorder(step.status)} bg-gray-900 flex items-center justify-center cursor-pointer transition-all relative z-10`}
                                whileHover={{ scale: 1.1 }}
                                animate={step.status === "active" ? {
                                    boxShadow: ["0 0 20px rgba(59, 130, 246, 0.3)", "0 0 40px rgba(59, 130, 246, 0.5)", "0 0 20px rgba(59, 130, 246, 0.3)"]
                                } : {}}
                                transition={{ duration: 1.5, repeat: step.status === "active" ? Infinity : 0 }}
                            >
                                <span className="text-2xl">{step.icon}</span>

                                {/* Status dot */}
                                <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${getStatusColor(step.status)}`} />
                            </motion.div>

                            {/* Label */}
                            <span className={`mt-3 text-xs font-medium ${step.status === "active" ? "text-blue-400" :
                                    step.status === "complete" ? "text-green-400" :
                                        "text-gray-500"
                                }`}>
                                {step.label}
                            </span>

                            {/* Tooltip on hover */}
                            {hoveredStep === step.id && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="absolute top-20 mt-2 w-40 p-3 bg-gray-800 border border-gray-700 rounded-xl shadow-xl z-20"
                                >
                                    <p className="text-xs text-gray-300">{step.description}</p>
                                </motion.div>
                            )}
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Result Summary (shown when complete) */}
            {result && currentStep >= 5 && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-8 p-4 bg-gray-900/50 border border-gray-800 rounded-xl"
                >
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className={`w-3 h-3 rounded-full ${result.prediction === "HATEFUL" ? "bg-red-500" : "bg-green-500"
                                }`} />
                            <span className="text-sm text-gray-400">
                                Decision: <span className={`font-medium ${result.prediction === "HATEFUL" ? "text-red-400" : "text-green-400"
                                    }`}>{result.prediction}</span>
                            </span>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-500">
                            <span className="flex items-center gap-1">
                                <span className={`w-1.5 h-1.5 rounded-full ${result.ragAvailable ? "bg-green-500" : "bg-gray-600"}`} />
                                RAG
                            </span>
                            <span className="flex items-center gap-1">
                                <span className={`w-1.5 h-1.5 rounded-full ${result.llmAvailable ? "bg-green-500" : "bg-gray-600"}`} />
                                LLM
                            </span>
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}
