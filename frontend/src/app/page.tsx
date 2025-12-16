"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui";

export default function Home() {
  const [inputText, setInputText] = useState("");
  const router = useRouter();

  const handleAnalyze = () => {
    if (inputText.trim()) {
      router.push(`/analyze?text=${encodeURIComponent(inputText)}`);
    }
  };

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#0B0F19] via-[#111827] to-[#0B0F19]" />

      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute w-96 h-96 bg-blue-600/20 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          style={{ top: "10%", left: "10%" }}
        />
        <motion.div
          className="absolute w-96 h-96 bg-red-600/10 rounded-full blur-3xl"
          animate={{
            x: [0, -50, 0],
            y: [0, 100, 0],
          }}
          transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          style={{ bottom: "10%", right: "10%" }}
        />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 py-20">
        <div className="grid lg:grid-cols-2 gap-16 items-center min-h-[80vh]">
          {/* Left: Hero Content */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6"
            >
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm text-blue-300">Research-Grade AI Detection</span>
            </motion.div>

            <h1 className="text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
              Explainable{" "}
              <span className="bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
                Hate Speech
              </span>{" "}
              Detection
            </h1>

            <p className="text-xl text-gray-400 mb-8 leading-relaxed">
              Context-Aware Retrieval Augmented Generation for{" "}
              <span className="text-white font-medium">accurate</span> and{" "}
              <span className="text-white font-medium">fair</span> content moderation.
            </p>

            {/* Quick Demo Input */}
            <div className="glass-card p-1 mb-8">
              <div className="flex flex-col sm:flex-row gap-3">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Type a sentence to analyze..."
                  className="flex-1 bg-transparent border-none outline-none text-white placeholder-gray-500 resize-none p-4 min-h-[60px]"
                  rows={2}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleAnalyze();
                    }
                  }}
                />
                <Button
                  onClick={handleAnalyze}
                  size="lg"
                  className="sm:self-end whitespace-nowrap"
                  disabled={!inputText.trim()}
                >
                  Analyze →
                </Button>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-6">
              {[
                { value: "610+", label: "Hate Indicators" },
                { value: "16", label: "Categories" },
                { value: "RAG", label: "Powered" },
              ].map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + i * 0.1 }}
                  className="text-center"
                >
                  <div className="text-2xl font-bold text-white">{stat.value}</div>
                  <div className="text-sm text-gray-500">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Right: Visualization */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="hidden lg:block"
          >
            <div className="relative">
              {/* Neural network visualization placeholder */}
              <div className="glass-card p-8 aspect-square flex items-center justify-center">
                <div className="relative w-full h-full">
                  {/* Center node */}
                  <motion.div
                    className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-700 rounded-full flex items-center justify-center shadow-lg shadow-blue-500/30"
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <span className="text-white font-bold text-sm">CAD-RAG</span>
                  </motion.div>

                  {/* Orbiting nodes */}
                  {["Context", "ML Model", "RAG", "LLM"].map((label, i) => {
                    const angle = (i * 90) * (Math.PI / 180);
                    const radius = 120;
                    const x = Math.cos(angle) * radius;
                    const y = Math.sin(angle) * radius;

                    return (
                      <motion.div
                        key={label}
                        className="absolute top-1/2 left-1/2 w-16 h-16 bg-gray-800/80 border border-gray-700 rounded-full flex items-center justify-center text-xs text-gray-300"
                        style={{
                          x: x - 32,
                          y: y - 32,
                        }}
                        animate={{
                          opacity: [0.5, 1, 0.5],
                        }}
                        transition={{
                          duration: 2,
                          delay: i * 0.3,
                          repeat: Infinity,
                        }}
                      >
                        {label}
                      </motion.div>
                    );
                  })}

                  {/* Connection lines */}
                  <svg className="absolute inset-0 w-full h-full" style={{ zIndex: -1 }}>
                    {[0, 1, 2, 3].map((i) => {
                      const angle = (i * 90) * (Math.PI / 180);
                      const radius = 120;
                      const x = Math.cos(angle) * radius + 150;
                      const y = Math.sin(angle) * radius + 150;

                      return (
                        <motion.line
                          key={i}
                          x1="50%"
                          y1="50%"
                          x2={x}
                          y2={y}
                          stroke="rgba(59, 130, 246, 0.3)"
                          strokeWidth="2"
                          strokeDasharray="5,5"
                          initial={{ pathLength: 0 }}
                          animate={{ pathLength: 1 }}
                          transition={{ duration: 1, delay: i * 0.2 }}
                        />
                      );
                    })}
                  </svg>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-20"
        >
          <h2 className="text-2xl font-bold text-white text-center mb-12">
            Why CAD-RAG?
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                icon: "🎯",
                title: "Context-Aware",
                description: "Retrieves relevant context to understand nuanced language and evolving terminology.",
              },
              {
                icon: "🔍",
                title: "Explainable",
                description: "Provides detailed rationale for each classification with evidence from knowledge bases.",
              },
              {
                icon: "⚖️",
                title: "Fair & Ethical",
                description: "Reduces bias through multi-source validation and expert model aggregation.",
              },
            ].map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 + i * 0.1 }}
                className="glass-card p-6 hover:border-blue-500/30 transition-colors"
              >
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </main>
  );
}
