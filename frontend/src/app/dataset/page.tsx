"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Card } from "@/components/ui";

export default function DatasetPage() {
    const datasetStats = {
        totalSamples: 159571,
        hateCategories: 16,
        indicators: 610,
        languages: 1,
    };

    const categoryData = [
        { name: "Religion-based hate", count: 12450, percentage: 7.8, color: "bg-red-500" },
        { name: "Race-based hate", count: 18320, percentage: 11.5, color: "bg-orange-500" },
        { name: "Gender-based hate", count: 15680, percentage: 9.8, color: "bg-yellow-500" },
        { name: "LGBTQ+ hate", count: 8940, percentage: 5.6, color: "bg-green-500" },
        { name: "Political hate", count: 21450, percentage: 13.4, color: "bg-blue-500" },
        { name: "Cyberbullying", count: 25670, percentage: 16.1, color: "bg-purple-500" },
        { name: "Other", count: 56061, percentage: 35.1, color: "bg-gray-500" },
    ];

    const labelDistribution = [
        { name: "toxic", count: 15294, percentage: 9.6 },
        { name: "severe_toxic", count: 1595, percentage: 1.0 },
        { name: "obscene", count: 8449, percentage: 5.3 },
        { name: "threat", count: 478, percentage: 0.3 },
        { name: "insult", count: 7877, percentage: 4.9 },
        { name: "identity_hate", count: 1405, percentage: 0.9 },
    ];

    const fairnessMetrics = [
        { group: "Religious Groups", falsePositive: 3.2, falseNegative: 5.1, accuracy: 91.7 },
        { group: "Ethnic Groups", falsePositive: 2.8, falseNegative: 4.6, accuracy: 92.6 },
        { group: "Gender Groups", falsePositive: 2.1, falseNegative: 3.8, accuracy: 94.1 },
        { group: "Political Groups", falsePositive: 4.5, falseNegative: 6.2, accuracy: 89.3 },
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
                        <Link href="/dataset" className="text-blue-400 text-sm font-medium">Dataset</Link>
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
                    <h1 className="text-4xl font-bold text-white mb-4">Dataset & Fairness</h1>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Transparency in our training data and fairness metrics for responsible AI.
                    </p>
                </motion.div>

                {/* Stats Grid */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12"
                >
                    {[
                        { label: "Total Samples", value: datasetStats.totalSamples.toLocaleString(), icon: "📊" },
                        { label: "Hate Categories", value: datasetStats.hateCategories, icon: "🏷️" },
                        { label: "Indicators", value: `${datasetStats.indicators}+`, icon: "🔍" },
                        { label: "Languages", value: datasetStats.languages, icon: "🌐" },
                    ].map((stat, i) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.2 + i * 0.05 }}
                        >
                            <Card variant="glass" className="text-center">
                                <div className="text-3xl mb-2">{stat.icon}</div>
                                <div className="text-2xl font-bold text-white">{stat.value}</div>
                                <div className="text-sm text-gray-400">{stat.label}</div>
                            </Card>
                        </motion.div>
                    ))}
                </motion.div>

                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Category Distribution */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.3 }}
                    >
                        <Card variant="glass">
                            <h2 className="text-xl font-semibold text-white mb-6">Category Distribution</h2>
                            <div className="space-y-4">
                                {categoryData.map((category, i) => (
                                    <motion.div
                                        key={category.name}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: 0.4 + i * 0.05 }}
                                    >
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-sm text-gray-300">{category.name}</span>
                                            <span className="text-sm text-gray-400">
                                                {category.count.toLocaleString()} ({category.percentage}%)
                                            </span>
                                        </div>
                                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                                            <motion.div
                                                className={`h-full ${category.color} rounded-full`}
                                                initial={{ width: 0 }}
                                                animate={{ width: `${category.percentage * 2.5}%` }}
                                                transition={{ delay: 0.5 + i * 0.05, duration: 0.5 }}
                                            />
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        </Card>
                    </motion.div>

                    {/* Label Distribution (Pie Chart Simulation) */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.3 }}
                    >
                        <Card variant="glass">
                            <h2 className="text-xl font-semibold text-white mb-6">Label Distribution</h2>
                            <div className="flex items-center justify-center mb-6">
                                {/* Simulated Pie Chart */}
                                <div className="relative w-48 h-48">
                                    <svg viewBox="0 0 100 100" className="transform -rotate-90">
                                        {labelDistribution.map((label, i) => {
                                            const colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E", "#3B82F6", "#A855F7"];
                                            const offset = labelDistribution.slice(0, i).reduce((sum, l) => sum + l.percentage * 4.5, 0);
                                            return (
                                                <motion.circle
                                                    key={label.name}
                                                    cx="50"
                                                    cy="50"
                                                    r="40"
                                                    fill="transparent"
                                                    stroke={colors[i]}
                                                    strokeWidth="20"
                                                    strokeDasharray={`${label.percentage * 4.5} ${100 * Math.PI}`}
                                                    strokeDashoffset={-offset}
                                                    initial={{ opacity: 0 }}
                                                    animate={{ opacity: 1 }}
                                                    transition={{ delay: 0.5 + i * 0.1 }}
                                                />
                                            );
                                        })}
                                    </svg>
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <div className="text-center">
                                            <div className="text-2xl font-bold text-white">22%</div>
                                            <div className="text-xs text-gray-400">Labeled</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                {labelDistribution.map((label, i) => {
                                    const colors = ["bg-red-500", "bg-orange-500", "bg-yellow-500", "bg-green-500", "bg-blue-500", "bg-purple-500"];
                                    return (
                                        <div key={label.name} className="flex items-center gap-2 text-sm">
                                            <div className={`w-3 h-3 rounded-full ${colors[i]}`} />
                                            <span className="text-gray-400">{label.name}</span>
                                            <span className="text-gray-500 ml-auto">{label.percentage}%</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </Card>
                    </motion.div>
                </div>

                {/* Fairness Metrics */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="mt-8"
                >
                    <Card variant="gradient-border">
                        <h2 className="text-xl font-semibold text-white mb-6">Fairness & Bias Indicators</h2>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b border-gray-800">
                                        <th className="text-left py-3 px-4 text-gray-400 font-medium">Group</th>
                                        <th className="text-center py-3 px-4 text-gray-400 font-medium">False Positive %</th>
                                        <th className="text-center py-3 px-4 text-gray-400 font-medium">False Negative %</th>
                                        <th className="text-center py-3 px-4 text-gray-400 font-medium">Accuracy %</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {fairnessMetrics.map((metric, i) => (
                                        <motion.tr
                                            key={metric.group}
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            transition={{ delay: 0.6 + i * 0.05 }}
                                            className="border-b border-gray-800/50"
                                        >
                                            <td className="py-4 px-4 text-white">{metric.group}</td>
                                            <td className="py-4 px-4 text-center">
                                                <span className={`px-2 py-1 rounded text-sm ${metric.falsePositive < 3 ? "bg-green-500/20 text-green-400" :
                                                        metric.falsePositive < 4 ? "bg-yellow-500/20 text-yellow-400" :
                                                            "bg-red-500/20 text-red-400"
                                                    }`}>
                                                    {metric.falsePositive}%
                                                </span>
                                            </td>
                                            <td className="py-4 px-4 text-center">
                                                <span className={`px-2 py-1 rounded text-sm ${metric.falseNegative < 4 ? "bg-green-500/20 text-green-400" :
                                                        metric.falseNegative < 5 ? "bg-yellow-500/20 text-yellow-400" :
                                                            "bg-red-500/20 text-red-400"
                                                    }`}>
                                                    {metric.falseNegative}%
                                                </span>
                                            </td>
                                            <td className="py-4 px-4 text-center">
                                                <span className="px-2 py-1 rounded text-sm bg-blue-500/20 text-blue-400">
                                                    {metric.accuracy}%
                                                </span>
                                            </td>
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        <p className="mt-4 text-sm text-gray-500">
                            * Metrics based on validation set. Lower false positive/negative rates indicate better fairness.
                        </p>
                    </Card>
                </motion.div>
            </div>
        </main>
    );
}
