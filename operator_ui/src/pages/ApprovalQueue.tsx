import {useEffect, useState} from "react";
import {decide, getJson} from "../api";

type Approval = {id:string; agent:string; risk:string; action:Record<string, unknown>};
export function ApprovalQueue() {
  const [items, setItems] = useState<Approval[]>([]);
  const load = () => getJson<Approval[]>("/approvals").then(setItems);
  useEffect(load, []);
  return <section><h2>Approval queue</h2>{items.map(item => <article key={item.id}><b>{item.risk}</b> {item.agent}<pre>{JSON.stringify(item.action, null, 2)}</pre><button onClick={() => decide(item.id, "approved").then(load)}>Approve</button><button onClick={() => decide(item.id, "rejected").then(load)}>Reject</button></article>)}</section>;
}

