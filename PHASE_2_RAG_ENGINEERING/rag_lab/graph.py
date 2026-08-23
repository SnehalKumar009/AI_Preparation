"""Graph RAG: retrieve by walking an entity graph (notebook 16).

An LLM extracts ``subject | relation | object`` triples from each chunk; we build a
directed graph from them (networkx) and remember which chunks mention each entity.
At query time we find the entities named in the question, expand to their graph
neighborhood, and return the chunks attached to those nodes — surfacing facts that
are connected rather than merely similar. Extraction runs one call per chunk, so it
defaults to Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from study_buddy import get_provider

from rag_lab.chunking import Chunk
from rag_lab.vectorstore import Hit

_EXTRACT = (
    "Extract factual relationships from the TEXT as triples. Output one triple per "
    "line in the form:\n  subject | relation | object\n"
    "Use short entity names and only information stated in the text. No extra "
    "commentary.\n\nTEXT:\n{text}\n\nTriples:"
)


@dataclass
class Triple:
    subject: str
    relation: str
    object: str


def _parse_triples(text: str) -> list[Triple]:
    triples: list[Triple] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append(Triple(*parts))
    return triples


class GraphRAG:
    """Extract an entity graph with an LLM, then retrieve by graph traversal."""

    def __init__(self, provider: str = "ollama", model: str | None = None):
        import networkx as nx  # lazy: only needed from notebook 16 onward

        self.provider = provider
        self.model = model
        self.graph = nx.DiGraph()
        self._chunks: list[Chunk] = []
        self._node_chunks: dict[str, set[int]] = {}
        self._llm = None

    def _extract(self, text: str) -> list[Triple]:
        if self._llm is None:
            self._llm = get_provider(self.provider)
        resp = self._llm.chat(
            [{"role": "user", "content": _EXTRACT.format(text=text)}],
            model=self.model,
            temperature=0.0,
        )
        return _parse_triples(resp.text)

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            print("---------------------------------")
            print(chunk.text)
            print(chunk.meta)
            ci = len(self._chunks)
            self._chunks.append(chunk)
            print("---------------------------------")
            print(f"Processing chunk {ci}: {chunk.text}")
            for t in self._extract(chunk.text):
                print("===================================================================")
                print(f"{t.subject} | {t.relation} | {t.object}")
                subj, obj = t.subject.lower().strip(), t.object.lower().strip()
                print("===================================================================")
                for key, label in ((subj, t.subject), (obj, t.object)):
                    print("########################################################################")
                    print(f"Adding node: {key} -> {label.strip()}")
                    self.graph.add_node(key, label=label.strip())
                    self._node_chunks.setdefault(key, set()).add(ci)
                    print(self._node_chunks)
                    print("########################################################################")
                self.graph.add_edge(subj, obj, relation=t.relation.strip())

    def _neighborhood(self, query: str, depth: int) -> set[str]:
        print(f"DEBUG: Finding neighborhood for query '{query}' with depth {depth}")
        q = query.lower()
        nodes = {n for n in self.graph.nodes if self.graph.nodes[n]["label"].lower() in q}
        print(f"DEBUG: Found initial nodes: {nodes}")
        frontier = set(nodes)
        print(f"DEBUG: Initial frontier: {frontier}")
        for i in range(depth):
            print(f"DEBUG: Expanding at depth {i}")
            nxt: set[str] = set()
            for n in frontier:
                successors = set(self.graph.successors(n))
                predecessors = set(self.graph.predecessors(n))
                print(f"DEBUG: Node {n} has successors: {successors}, predecessors: {predecessors}")
                nxt |= successors | predecessors
            nodes |= nxt
            frontier = nxt
        return nodes

    def edges_near(self, query: str, depth: int = 1) -> list[tuple[str, str, str]]:
        """Human-readable (subject, relation, object) edges near the query entities."""
        nodes = self._neighborhood(query, depth)
        return [
            (self.graph.nodes[u]["label"], d.get("relation", ""), self.graph.nodes[v]["label"])
            for u, v, d in self.graph.edges(data=True)
            if u in nodes or v in nodes
        ]

    def search(self, query: str, k: int = 4, depth: int = 1) -> list[Hit]:
        """Retrieve chunks attached to entities near the query, ranked by overlap."""
        nodes = self._neighborhood(query, depth)
        scores: dict[int, int] = {}
        for n in nodes:
            for ci in self._node_chunks.get(n, ()):
                scores[ci] = scores.get(ci, 0) + 1
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [Hit(self._chunks[ci], float(s)) for ci, s in ranked[:k]]

    def __len__(self) -> int:
        return self.graph.number_of_nodes()
