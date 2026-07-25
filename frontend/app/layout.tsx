import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'PulmoSight | Advanced Chest X-Ray AI Platform',
  description: 'Production-grade medical intelligence platform for Pneumonia detection using PyTorch, Grad-CAM, and LLM clinical reporting.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
