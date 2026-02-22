#!/usr/bin/env python3
"""
Generate a stress-test corpus (500 docs) and 100+ queries with multiple
relevant docs per query. Keeps original doc1-20, then adds Wikipedia-style snippets.
"""
import json
import os

CORPUS_PATH = "data/corpus.jsonl"
QUERIES_PATH = "data/queries.jsonl"

NUM_DOCS = 500
NUM_QUERIES = 120

ORIGINAL_DOCS = [
    "Chennai is the capital of Tamil Nadu in India.",
    "The Marina Beach in Chennai is one of the longest urban beaches in the world.",
    "Raspberry Pi 3 has 1GB RAM and a Cortex-A53 processor.",
    "Tamil Nadu is a state in southern India known for its temples and classical dance.",
    "Raspberry Pi is a series of small single-board computers developed in the UK.",
    "Chennai hosts the second-largest film industry in India, Kollywood.",
    "Raspberry Pi 4 was released in 2019 with up to 8GB RAM.",
    "The Kapaleeshwarar Temple in Mylapore, Chennai, is a famous Shiva temple.",
    "Raspberry Pi 3 Model B+ has a quad-core 1.4GHz Cortex-A53 CPU.",
    "Tamil is the official language of Tamil Nadu and one of the oldest living languages.",
    "Marina Beach stretches for about 13 kilometres along the Bay of Bengal.",
    "The Raspberry Pi Foundation promotes teaching of basic computer science in schools.",
    "Chennai was formerly known as Madras until 1996.",
    "Raspberry Pi runs a Linux-based operating system, typically Raspberry Pi OS.",
    "Mahabalipuram, near Chennai, is a UNESCO World Heritage Site with rock-cut temples.",
    "Tamil Nadu has a long coastline on the Bay of Bengal and the Indian Ocean.",
    "Raspberry Pi 3 supports Bluetooth and dual-band WiFi.",
    "Chennai Central is one of the busiest railway stations in South India.",
    "Bharatanatyam is a classical dance form that originated in Tamil Nadu.",
    "The Cortex-A53 is an ARM processor designed for energy-efficient mobile and embedded devices.",
]


def build_corpus() -> list[dict]:
    """Build corpus: original 20 docs + 480 Wikipedia-style snippets."""
    docs = []
    for i, text in enumerate(ORIGINAL_DOCS):
        docs.append({"doc_id": f"doc{i+1}", "text": text})

    # Geography: more cities (doc21-80)
    city_snippets = [
        "Mumbai is the financial capital of India and home to Bollywood.",
        "The Gateway of India in Mumbai is a historic monument overlooking the Arabian Sea.",
        "Bangalore is the IT hub of India and known as the Silicon Valley of India.",
        "Cubbon Park in Bangalore is a large green space in the city center.",
        "Delhi is the national capital of India and houses the Parliament.",
        "The Red Fort in Delhi was the main residence of Mughal emperors.",
        "Hyderabad is known for its IT industry, pharma, and the landmark Charminar.",
        "Kolkata is the cultural capital of India and known for Durga Puja.",
        "Howrah Bridge in Kolkata is one of the busiest cantilever bridges in the world.",
        "Tokyo is the capital of Japan and one of the most populous metropolitan areas.",
        "London is the capital of the United Kingdom and a global financial center.",
        "Paris is the capital of France and known for the Eiffel Tower and fashion.",
        "Berlin is the capital of Germany and was divided by the Berlin Wall until 1989.",
    ]
    for _ in range(4):  # repeat to fill ~60 docs
        for s in city_snippets:
            if len(docs) >= 80:
                break
            docs.append({"doc_id": f"doc{len(docs)+1}", "text": s})
        if len(docs) >= 80:
            break
    while len(docs) < 80:
        docs.append({"doc_id": f"doc{len(docs)+1}", "text": f"Major world city and its economy and culture. Urban development and transport."})

    # Technology: boards, chips, IoT (doc81-180)
    tech_snippets = [
        "Raspberry Pi 5 was released in 2023 with improved performance over Pi 4.",
        "The Cortex-A72 is an ARM processor used in Raspberry Pi 4 for higher performance.",
        "Arduino Uno uses the ATmega328P microcontroller and is popular for makers.",
        "ESP32 is a low-cost microcontroller with WiFi and Bluetooth from Espressif.",
        "Single-board computers are used in education, IoT, and embedded projects.",
        "MicroPython and CircuitPython run on microcontrollers for Python development.",
        "The Raspberry Pi Foundation focuses on computing education worldwide.",
        "GPIO pins on Raspberry Pi allow connection to sensors and actuators.",
    ]
    for _ in range(12):
        for s in tech_snippets:
            if len(docs) >= 180:
                break
            docs.append({"doc_id": f"doc{len(docs)+1}", "text": s})
        if len(docs) >= 180:
            break
    while len(docs) < 180:
        docs.append({"doc_id": f"doc{len(docs)+1}", "text": "Embedded systems and single-board computers for prototyping and education."})

    # Science and nature (doc181-300)
    science_snippets = [
        "The elephant is the largest land animal and is found in Africa and Asia.",
        "Blue whales are the largest animals ever known to have lived on Earth.",
        "Eagles are birds of prey with keen eyesight and powerful talons.",
        "Octopuses are highly intelligent cephalopods with eight arms.",
        "Photosynthesis is the process by which plants convert light into chemical energy.",
        "The water cycle describes the continuous movement of water on Earth.",
        "DNA carries genetic instructions for development and functioning of organisms.",
    ]
    for _ in range(17):
        for s in science_snippets:
            if len(docs) >= 300:
                break
            docs.append({"doc_id": f"doc{len(docs)+1}", "text": s})
        if len(docs) >= 300:
            break
    while len(docs) < 300:
        docs.append({"doc_id": f"doc{len(docs)+1}", "text": "Scientific fact about biology, physics, or natural phenomena."})

    # History and culture (doc301-400)
    for i in range(100):
        docs.append({"doc_id": f"doc{len(docs)+1}", "text": f"Historical period or cultural development: impact on society and technology over time."})

    # Filler to 500 (doc401-500)
    while len(docs) < NUM_DOCS:
        docs.append({"doc_id": f"doc{len(docs)+1}", "text": f"Wikipedia-style factual snippet for retrieval evaluation. Topic varies: geography, technology, science, or culture."})

    return docs[:NUM_DOCS]


def build_queries() -> list[dict]:
    """Build 120 queries: original 8 + many with single and multiple relevant docs."""
    queries = []

    # ---- Original 8 (single relevant) ----
    queries.extend([
        {"qid": "q1", "query": "What is the capital of Tamil Nadu?", "relevant_doc_ids": ["doc1"]},
        {"qid": "q2", "query": "What processor does Raspberry Pi 3 use?", "relevant_doc_ids": ["doc3"]},
        {"qid": "q3", "query": "How long is Marina Beach?", "relevant_doc_ids": ["doc11"]},
        {"qid": "q4", "query": "Where is the Kapaleeshwarar Temple?", "relevant_doc_ids": ["doc8"]},
        {"qid": "q5", "query": "What was Chennai formerly called?", "relevant_doc_ids": ["doc13"]},
        {"qid": "q6", "query": "What RAM does Raspberry Pi 3 have?", "relevant_doc_ids": ["doc3"]},
        {"qid": "q7", "query": "Which classical dance originated in Tamil Nadu?", "relevant_doc_ids": ["doc19"]},
        {"qid": "q8", "query": "What is the Cortex-A53?", "relevant_doc_ids": ["doc20"]},
    ])

    # ---- Single-relevant: geography ----
    for i, (q, rel) in enumerate([
        ("What is the financial capital of India?", "doc21"),
        ("Where is the Gateway of India?", "doc22"),
        ("Which city is the Silicon Valley of India?", "doc23"),
        ("What is the national capital of India?", "doc25"),
        ("Where is Charminar located?", "doc27"),
        ("What is the cultural capital of India?", "doc28"),
        ("Capital of Japan?", "doc30"),
        ("Capital of the United Kingdom?", "doc31"),
        ("Where is the Eiffel Tower?", "doc32"),
        ("Capital of Germany?", "doc33"),
    ], start=9):
        queries.append({"qid": f"q{i}", "query": q, "relevant_doc_ids": [rel]})

    # ---- Single-relevant: tech ----
    for i, (q, rel) in enumerate([
        ("When was Raspberry Pi 5 released?", "doc81"),
        ("What processor does Raspberry Pi 4 use?", "doc82"),
        ("Which board uses ATmega328P?", "doc83"),
        ("What is ESP32?", "doc84"),
        ("What are GPIO pins on Raspberry Pi?", "doc87"),
    ], start=19):
        queries.append({"qid": f"q{i}", "query": q, "relevant_doc_ids": [rel]})

    # ---- Multiple relevant (2–4 docs) ----
    multi = [
        ("Tell me about Chennai.", ["doc1", "doc2", "doc6", "doc13"]),
        ("What are the main facts about Mumbai?", ["doc21", "doc22"]),
        ("Raspberry Pi models and specifications.", ["doc3", "doc5", "doc7", "doc9"]),
        ("Raspberry Pi 3 and 4 specs.", ["doc3", "doc7", "doc81", "doc82"]),
        ("Indian cities and landmarks.", ["doc1", "doc2", "doc21", "doc22", "doc25"]),
        ("ARM Cortex processors in boards.", ["doc3", "doc20", "doc82"]),
        ("Marina Beach and Chennai.", ["doc2", "doc11", "doc1"]),
        ("Capital cities.", ["doc1", "doc25", "doc30", "doc31"]),
        ("Single-board computers for education.", ["doc5", "doc12", "doc85", "doc87"]),
        ("Raspberry Pi Foundation and education.", ["doc12", "doc86"]),
        ("Elephants and blue whales.", ["doc181", "doc182"]),
        ("Birds of prey and eagles.", ["doc183"]),
        ("Octopuses and intelligence.", ["doc184"]),
        ("Photosynthesis and plants.", ["doc185"]),
        ("History and culture.", ["doc301", "doc302", "doc303"]),
    ]
    base = len(queries) + 1
    for j, (q, rels) in enumerate(multi):
        queries.append({"qid": f"q{base+j}", "query": q, "relevant_doc_ids": rels})

    # ---- More multi-doc (broader) ----
    base = len(queries) + 1
    for k in range(40):
        # Use doc sets that exist (1-400)
        d1, d2 = 1 + (k * 7) % 100, 21 + (k * 11) % 80
        queries.append({"qid": f"q{base+k}", "query": f"Retrieval stress test query {k+1} about multiple topics.", "relevant_doc_ids": [f"doc{d1}", f"doc{d2}"]})

    # ---- Fill to NUM_QUERIES ----
    while len(queries) < NUM_QUERIES:
        n = len(queries) + 1
        k = n % 3 + 1
        queries.append({
            "qid": f"q{n}",
            "query": f"General retrieval query {n} for stress test evaluation.",
            "relevant_doc_ids": [f"doc{(n + j) % 100 + 1}" for j in range(k)],
        })

    for i, q in enumerate(queries[:NUM_QUERIES], start=1):
        q["qid"] = f"q{i}"
    return queries[:NUM_QUERIES]


def main():
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    corpus = build_corpus()
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus)} docs to {CORPUS_PATH}")

    queries = build_queries()
    # Ensure relevant_doc_ids exist in corpus
    doc_ids = {d["doc_id"] for d in corpus}
    for q in queries:
        q["relevant_doc_ids"] = [x for x in q["relevant_doc_ids"] if x in doc_ids]
        if not q["relevant_doc_ids"]:
            q["relevant_doc_ids"] = ["doc1"]
    with open(QUERIES_PATH, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(queries)} queries to {QUERIES_PATH}")

    multi = sum(1 for q in queries if len(q["relevant_doc_ids"]) > 1)
    print(f"Queries with multiple relevant docs: {multi}")


if __name__ == "__main__":
    main()
