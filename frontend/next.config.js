/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/v1/**',
      },
      {
        protocol: 'http',
        hostname: 'backend',
        port: '8000',
        pathname: '/api/v1/**',
      },
    ],
    unoptimized: true,
  },
  // Suppress React hydration warnings in dev
  reactStrictMode: true,
};

module.exports = nextConfig;
