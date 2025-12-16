import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CAD-RAG | Explainable Hate Speech Detection",
  description: "Context-Aware Retrieval Augmented Generation for accurate and fair content moderation.",
  keywords: ["hate speech detection", "RAG", "explainable AI", "content moderation", "NLP"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} antialiased bg-[#0B0F19] text-gray-100 min-h-screen`}>
        {children}
      </body>
    </html>
  );
}
