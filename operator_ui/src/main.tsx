import React from "react"; import {createRoot} from "react-dom/client"; import {ApprovalQueue} from "./pages/ApprovalQueue"; import {TraceViewer} from "./pages/TraceViewer"; import {Incidents} from "./pages/Incidents"; import {Routing} from "./pages/Routing"; import "./style.css";
function App(){return <><header><h1>AI Systems Operator</h1><p>Approvals, traces, incidents, and model routing.</p></header><main><ApprovalQueue/><TraceViewer/><Incidents/><Routing/></main></>};createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);

