"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface CardProps {
    children: ReactNode;
    className?: string;
    variant?: "default" | "glass" | "gradient-border";
    animate?: boolean;
}

export default function Card({
    children,
    className = "",
    variant = "default",
    animate = true
}: CardProps) {
    const variants = {
        default: "bg-gray-900/50 border border-gray-800",
        glass: "glass-card",
        "gradient-border": "gradient-border bg-gray-900/80",
    };

    const Component = animate ? motion.div : "div";
    const animationProps = animate ? {
        initial: { opacity: 0, y: 20 },
        animate: { opacity: 1, y: 0 },
        transition: { duration: 0.4 }
    } : {};

    return (
        <Component
            className={`rounded-2xl p-6 ${variants[variant]} ${className}`}
            {...animationProps}
        >
            {children}
        </Component>
    );
}
