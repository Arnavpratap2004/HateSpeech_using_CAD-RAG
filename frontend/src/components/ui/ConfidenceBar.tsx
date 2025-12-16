"use client";

import { motion } from "framer-motion";

interface ConfidenceBarProps {
    value: number; // 0-100
    label?: string;
    colorScheme?: "danger" | "success" | "warning" | "blue";
}

export default function ConfidenceBar({
    value,
    label,
    colorScheme = "blue"
}: ConfidenceBarProps) {
    const colors = {
        danger: "bg-gradient-to-r from-red-600 to-red-400",
        success: "bg-gradient-to-r from-green-600 to-green-400",
        warning: "bg-gradient-to-r from-yellow-600 to-yellow-400",
        blue: "bg-gradient-to-r from-blue-600 to-blue-400",
    };

    // Auto-select color based on value if not specified
    const autoColor = value >= 70 ? "danger" : value >= 40 ? "warning" : "success";
    const fillColor = colorScheme === "blue" ? colors.blue : colors[autoColor];

    return (
        <div className="w-full">
            {label && (
                <div className="flex justify-between items-center mb-1">
                    <span className="text-sm text-gray-400">{label}</span>
                    <span className="text-sm font-medium text-white">{value.toFixed(0)}%</span>
                </div>
            )}
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <motion.div
                    className={`h-full rounded-full ${fillColor}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${value}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                />
            </div>
        </div>
    );
}
