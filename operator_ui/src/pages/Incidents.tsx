import {useEffect, useState} from "react"; import {getJson} from "../api";
type Incident={id:string;status:string;report:string};
export function Incidents(){const [items,setItems]=useState<Incident[]>([]);useEffect(()=>{getJson<Incident[]>("/incidents").then(setItems)},[]);return <section><h2>Incidents</h2>{items.map(x=><p key={x.id}>{x.id} — {x.status} — {x.report}</p>)}</section>}

