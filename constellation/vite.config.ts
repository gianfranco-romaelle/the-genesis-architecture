import { createReadStream, existsSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";
import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const genesisRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  base: "/constellation/",
  plugins: [
    react(),
    {
      name: "serve-genesis-json",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = req.url ?? "";
          if (url.endsWith(".json") && !url.includes("?")) {
            const candidate = path.join(genesisRoot, url.slice(1));
            if (existsSync(candidate)) {
              res.setHeader("Content-Type", "application/json");
              createReadStream(candidate).pipe(res);
              return;
            }
          }
          next();
        });
      },
    },
  ],
  server: {
    fs: {
      allow: [genesisRoot],
    },
  },
});
