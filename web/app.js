(function () {
  "use strict";

  const thread = document.getElementById("thread");
  const emptyState = document.getElementById("emptyState");
  const form = document.getElementById("composer");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const micBtn = document.getElementById("micBtn");
  const voiceNote = document.getElementById("voiceNote");
  const modeSelect = document.getElementById("modeSelect");
  const useLlmToggle = document.getElementById("useLlmToggle");
  const autoSpeakToggle = document.getElementById("autoSpeakToggle");

  const pillRag = document.getElementById("pillRag");
  const pillNlu = document.getElementById("pillNlu");
  const pillLlm = document.getElementById("pillLlm");

  function toSpeech(text) {
    return String(text)
      .replace(/https?:\/\/\S+/g, "")
      .replace(/\[[^\]]*\]/g, "")
      .replace(/\bPOL_[A-Z]+_\d+\b/g, "")
      .replace(/%\s?(\d+)/g, "yüzde $1")
      .replace(/(\d+)\s*-\s*(\d+)/g, "$1 ila $2")
      .replace(/\s*(TL|₺)\b/g, " lira")
      .replace(/[*_#`>~]/g, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  function speak(text) {
    if (!("speechSynthesis" in window)) return false;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(toSpeech(text));
    u.lang = "tr-TR";
    u.rate = 1.03;
    window.speechSynthesis.speak(u);
    return true;
  }

  async function loadHealth() {
    try {
      const r = await fetch("/api/health", { cache: "no-store" });
      if (!r.ok) throw new Error(r.status);
      const h = await r.json();
      pillRag.textContent = `RAG: ${h.policy_count} politika · ${h.rag_dense_backend}`;
      pillRag.className = "pill ok";
      pillNlu.textContent = `NLU: ${h.nlu_backend}`;
      pillNlu.className = "pill ok";
      if (h.ollama_available) {
        pillLlm.textContent = `LLM: ${h.ollama_model}`;
        pillLlm.className = "pill ok";
      } else {
        pillLlm.textContent = "LLM: Ollama yok (şablon aktif)";
        pillLlm.className = "pill warn";
      }
    } catch {
      pillRag.textContent = "Sunucu bağlantısı yok";
      pillRag.className = "pill warn";
    }
  }

  function appendUser(text) {
    if (emptyState) emptyState.remove();
    const div = document.createElement("div");
    div.className = "msg user";
    div.textContent = text;
    thread.appendChild(div);
    thread.scrollTop = thread.scrollHeight;
  }

  function appendAssistant(data) {
    const div = document.createElement("div");
    div.className = "msg assistant";

    const p = document.createElement("div");
    p.textContent = data.reply;
    div.appendChild(p);

    const actions = document.createElement("div");
    actions.className = "msg-actions";
    const speakBtn = document.createElement("button");
    speakBtn.type = "button";
    speakBtn.className = "speak-btn";
    speakBtn.textContent = "🔊 Dinle";
    speakBtn.addEventListener("click", () => speak(data.reply));
    actions.appendChild(speakBtn);

    const badge = document.createElement("span");
    badge.className = "muted";
    badge.textContent = `${data.policy.id || "—"} · ${data.backends.generation}`;
    actions.appendChild(badge);
    div.appendChild(actions);

    const details = document.createElement("details");
    details.className = "inspector";
    const summary = document.createElement("summary");
    summary.textContent = "Retrieval & NLU İncele";
    details.appendChild(summary);

    const grid = document.createElement("div");
    grid.className = "inspector-grid";
    const topEv = data.evidences && data.evidences[0];
    const lines = [
      `Politika: [${data.policy.id}] ${data.policy.title}`,
      `NLU: category=${data.nlu.category} (${data.nlu.category_confidence}), intent=${data.nlu.intent}, sentiment=${data.nlu.sentiment_score}`,
      topEv ? `Skorlar: bm25=${topEv.bm25_score} (rank ${topEv.bm25_rank}), dense=${topEv.dense_score} (rank ${topEv.dense_rank}), rrf=${topEv.rrf_score}` : "Kanıt yok",
      `Doğrulama: retrieval=${data.grounding.retrieval_verified}, numeric=${data.grounding.numeric_grounding_passed}`,
      `Süre (ms): nlu=${data.timings_ms.nlu}, retrieval=${data.timings_ms.retrieval}, gen=${data.timings_ms.generation}`,
      `Backend: dense=${data.backends.dense}, nlu=${data.backends.nlu}`,
    ];
    for (const line of lines) {
      const row = document.createElement("div");
      row.textContent = line;
      grid.appendChild(row);
    }
    details.appendChild(grid);
    div.appendChild(details);

    thread.appendChild(div);
    thread.scrollTop = thread.scrollHeight;

    if (autoSpeakToggle.checked) {
      speak(data.reply);
    }
  }

  async function sendMessage(text) {
    const clean = text.trim();
    if (!clean) return;
    appendUser(clean);
    input.value = "";
    sendBtn.disabled = true;
    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: clean,
          top_k: 3,
          retrieval_mode: modeSelect.value,
          use_llm: useLlmToggle.checked,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      appendAssistant(data);
    } catch (err) {
      appendAssistant({
        reply: `İstek başarısız oldu: ${err.message}`,
        policy: { id: "", title: "" },
        evidences: [],
        nlu: { category: "-", category_confidence: 0, intent: "-", sentiment_score: 0 },
        grounding: { retrieval_verified: false, numeric_grounding_passed: false },
        timings_ms: { nlu: 0, retrieval: 0, generation: 0 },
        backends: { dense: "-", nlu: "-", generation: "error" },
      });
    } finally {
      sendBtn.disabled = false;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });

  document.getElementById("samples").addEventListener("click", (e) => {
    const btn = e.target.closest(".sample-btn");
    if (!btn) return;
    sendMessage(btn.dataset.q || "");
  });

  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    micBtn.disabled = true;
    voiceNote.textContent = "Tarayıcınız Web Speech API (STT) desteklemiyor; metin girişi ve TTS kullanılabilir.";
  } else {
    let rec = null;
    let listening = false;
    micBtn.addEventListener("click", () => {
      if (listening && rec) {
        rec.stop();
        return;
      }
      rec = new SpeechRec();
      rec.lang = "tr-TR";
      rec.interimResults = true;
      rec.continuous = false;
      rec.onstart = () => {
        listening = true;
        micBtn.classList.add("listening");
        micBtn.textContent = "⏹️ Dinleniyor";
        voiceNote.textContent = "Türkçe konuşun…";
      };
      rec.onresult = (ev) => {
        let txt = "";
        for (let i = 0; i < ev.results.length; i++) {
          txt += ev.results[i][0].transcript;
        }
        input.value = txt.trim();
      };
      rec.onerror = (ev) => {
        voiceNote.textContent = `Mikrofon hatası: ${ev.error}`;
      };
      rec.onend = () => {
        listening = false;
        micBtn.classList.remove("listening");
        micBtn.textContent = "🎙️ Konuş";
        if (input.value.trim()) {
          voiceNote.textContent = "Ses metne dönüştürüldü; gönderebilirsiniz.";
        }
      };
      rec.start();
    });
  }

  loadHealth();
})();
