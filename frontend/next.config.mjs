/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Verification builds write to a separate directory so they can never
  // clobber a running `next dev` server's chunks (npm run build:check).
  distDir: process.env.NEXT_DIST_DIR || ".next",
  // The marketing page owns the root; the workbench lives at /workbench. Here
  // (not only vercel.json) so `next dev` redirects too — otherwise local and
  // production disagree about what "/" does.
  async redirects() {
    return [{ source: "/", destination: "/landing", permanent: false }];
  },
};

export default nextConfig;
