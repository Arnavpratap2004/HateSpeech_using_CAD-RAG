"use client";

interface BadgeProps {
    type: "hate" | "neutral" | "offensive" | "info";
    children: React.ReactNode;
    size?: "sm" | "md";
}

export default function Badge({ type, children, size = "md" }: BadgeProps) {
    const types = {
        hate: "bg-red-500/20 text-red-400 border-red-500/50",
        neutral: "bg-green-500/20 text-green-400 border-green-500/50",
        offensive: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50",
        info: "bg-blue-500/20 text-blue-400 border-blue-500/50",
    };

    const sizes = {
        sm: "px-2 py-0.5 text-xs",
        md: "px-3 py-1 text-sm",
    };

    return (
        <span className={`inline-flex items-center font-medium rounded-full border ${types[type]} ${sizes[size]}`}>
            {children}
        </span>
    );
}
