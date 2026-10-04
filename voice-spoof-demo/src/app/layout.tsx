import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice Spoof Demo",
  description: "Detect spoofed voices in real-time",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}