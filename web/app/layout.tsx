import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Human13 · Physiological Systems Research",
  description: "A research workspace for tracing physiological responses across molecules, cells, and whole-body systems. Hyunseung Kong · GC Biopharma.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
