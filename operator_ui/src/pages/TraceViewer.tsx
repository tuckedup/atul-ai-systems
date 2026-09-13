import {useState} from "react";
import {getJson} from "../api";

type Span = {name:string; start_time:number; end_time:number; attributes:Record<string, unknown>};
export function TraceViewer() {
  const [id, setId] = useState(""); const [spans, setSpans] = useState<Span[]>([]);
  return <section><h2>Trace viewer</h2><input value={id} onChange={e => setId(e.target.value)}/><button onClick={() => getJson<Span[]>(`/traces/${id}`).then(setSpans)}>Load</button><ol>{spans.map((span, i) => <li key={i}><b>{span.name}</b> {String(span.attributes["llm.model"] ?? "")} {String(span.attributes["aisys.latency_ms"] ?? "")}ms</li>)}</ol></section>;
}

