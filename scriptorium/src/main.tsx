import React from "react";
import ReactDOM from "react-dom/client";
import { ScriptoriumView } from "./features/scriptorium/scriptorium-view";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ScriptoriumView />
  </React.StrictMode>,
);
