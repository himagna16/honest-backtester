/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Verification builds write to a separate directory so they can never
  // clobber a running `next dev` server's chunks (npm run build:check).
  distDir: process.env.NEXT_DIST_DIR || ".next",
};

export default nextConfig;
