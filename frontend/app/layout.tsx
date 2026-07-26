import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'PulmoSight | Advanced Chest X-Ray AI Platform',
  description: 'Production-grade medical intelligence platform for Pneumonia detection using PyTorch, Grad-CAM, and LLM clinical reporting.',
  keywords: ['pneumonia detection', 'chest x-ray', 'AI', 'medical imaging', 'Grad-CAM', 'deep learning'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.className} antialiased`}>{children}</body>
    </html>
  );
}
