import type { Metadata } from "next";
import { Dongle, Inter } from "next/font/google";
import "./globals.css";

const dongle = Dongle({
  subsets: ["latin"],
  weight: ["300", "400"],
  variable: "--font-dongle",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Svara RAG | Voice-Enabled RAG Dashboard",
  description: "Minimal, production-grade Voice-Enabled RAG system designed by Zero_Day_Devs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${dongle.variable} font-sans antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

