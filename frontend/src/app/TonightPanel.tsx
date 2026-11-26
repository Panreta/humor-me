"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

const API = "http://localhost:8000";
const MAX_FACTS = 1; // mirrors MAX_TONIGHT_FACTS in backend/app/tonight.py

type Fetched = { enabled: boolean; market?: { line: string } | null; gists?: string[]; errors?: string[] };
export type Approved = { facts: string[]; market: string | null; gists: string[] };

// The pre-show panel (D-117): the market line, one checkbox per headline gist (all unchecked), and one optional line.
// Renders nothing when the backend has TONIGHT off or cannot be reached, so the show starts as it always did.
export default function TonightPanel({ onChange }: { onChange: (approved: Approved | undefined) => void }) {
  const [data, setData] = useState<Fetched | null>(null);
  const [useMarket, setUseMarket] = useState(true);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [fact, setFact] = useState("");
  const [verdict, setVerdict] = useState("");

  // The parent passes a new function on every render; calling it through a ref keeps the effect below from looping.
  const report = useRef(onChange);
  useEffect(() => {
    report.current = onChange;
  }, [onChange]);

  const load = useCallback((refresh: boolean) => {
    fetch(`${API}/tonight${refresh ? "?refresh=true" : ""}`)
      .then((r) => r.json())
      .then((d: Fetched) => {
        setData(d);
        setPicked(new Set());
      })
      .catch(() => setData(null));
  }, []);
  useEffect(() => load(false), [load]);

  useEffect(() => {
    if (!data?.enabled) return;
    const facts = fact.trim() ? [fact.trim()].slice(0, MAX_FACTS) : [];
    const market = useMarket && data.market ? data.market.line : null;
    const gists = (data.gists ?? []).filter((g) => picked.has(g));
    report.current(facts.length || market || gists.length ? { facts, market, gists } : undefined);
  }, [data, useMarket, picked, fact]);

  const check = () => {
    if (!fact.trim()) return;
    fetch(`${API}/tonight/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ facts: [fact.trim()] }),
    })
      .then((r) => r.json())
      .then((d: { verdicts: { ok: boolean; reason: string }[] }) => {
        const v = d.verdicts[0];
        setVerdict(v ? (v.ok ? "OK to use." : `Will be left out: ${v.reason}`) : "");
      })
      .catch(() => setVerdict("Could not check."));
  };

  if (!data?.enabled) return null;
  const toggle = (g: string) =>
    setPicked((s) => {
      const next = new Set(s);
      if (next.has(g)) next.delete(g);
      else next.add(g);
      return next;
    });

  return (
    <fieldset className={styles.tonight}>
      <legend>Tonight (optional)</legend>
      {data.market ? (
        <label>
          <input type="checkbox" checked={useMarket} onChange={(e) => setUseMarket(e.target.checked)} />
          {data.market.line}
        </label>
      ) : (
        <span>No market line today.</span>
      )}
      {(data.gists ?? []).map((g) => (
        <label key={g}>
          <input type="checkbox" checked={picked.has(g)} onChange={() => toggle(g)} />
          {g}
        </label>
      ))}
      {!data.gists?.length && <span>No headlines today.</span>}
      <div className={styles.factRow}>
        <input
          value={fact}
          maxLength={120}
          placeholder="One line about tonight, e.g. it's a hackathon (optional)"
          onChange={(e) => {
            setFact(e.target.value);
            setVerdict("");
          }}
        />
        <button type="button" className={styles.start} onClick={check} disabled={!fact.trim()}>
          Check
        </button>
      </div>
      {verdict && <span role="status">{verdict}</span>}
      <button type="button" className={styles.start} onClick={() => load(true)}>
        Fetch again
      </button>
    </fieldset>
  );
}
