import {useEffect, useState} from "react"; import {getJson} from "../api";
type Route={model:string;provider:string;quality:Record<string,number>;weight:number};
export function Routing(){const [items,setItems]=useState<Route[]>([]);useEffect(()=>{getJson<Route[]>("/routing").then(setItems)},[]);return <section><h2>Routing</h2><pre>{JSON.stringify(items,null,2)}</pre></section>}

