const API = import.meta.env.VITE_OPERATOR_API ?? "http://localhost:8000";
const token = import.meta.env.VITE_OPERATOR_TOKEN ?? "local-viewer-token";

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`, {headers: {Authorization: `Bearer ${token}`}});
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

export async function decide(id: string, verdict: "approved" | "rejected"): Promise<void> {
  const response = await fetch(`${API}/approvals/${id}/decision`, {method: "POST", headers: {Authorization: `Bearer ${token}`, "Content-Type": "application/json"}, body: JSON.stringify({verdict})});
  if (!response.ok) throw new Error(await response.text());
}

