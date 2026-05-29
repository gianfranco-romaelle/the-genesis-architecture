import React from "react";
import ReactDOM from "react-dom/client";
import { ConstellationView } from "./features/constellation/constellation-view";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConstellationView />
  </React.StrictMode>,
);
