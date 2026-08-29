"""
Voice -> Clean Text (Wispr Flow-style) — Gradio app.

Loads the two fine-tuned models pushed to the Hub by the training
notebook:
  - WHISPER_REPO_ID: verbatim, disfluency-robust ASR
  - LLM_REPO_ID:     cleanup + tone adaptation

Architecture recap (see the training notebook for why it's split this
way): ASR stays faithful to what was actually said; all rewriting
(removing filler words, fixing grammar, adapting tone) happens in the
LLM stage, where it's visible and controllable via the system prompt.
This keeps the pipeline honest — it can't silently put words in your
mouth during transcription.

UI notes
--------
This version presents the pipeline as an animated left-to-right flow
(Mic -> ASR -> Cleanup -> Output) and adds a single "talk key":

  - Press and HOLD the key (or the on-screen button) -> push-to-talk.
    Recording starts on press, stops (and auto-runs the pipeline) on
    release.
  - DOUBLE-CLICK / DOUBLE-TAP the key -> hands-free mode. Recording
    starts and keeps going without holding anything down. Click/tap
    the key once more (single click) to stop and run the pipeline.

The key is bound to the spacebar as well as a big on-screen circular
button, so it works with mouse, touch, or keyboard.
"""

import os
import time

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

# ---------------------------------------------------------------------------
# Config — point these at the repos the notebook pushed to
# ---------------------------------------------------------------------------

WHISPER_REPO_ID = os.environ.get("WHISPER_REPO_ID", "aijadugar/wisprflow-clone-whisper")
LLM_REPO_ID = os.environ.get("LLM_REPO_ID", "aijadugar/wisprflow-clone-llm")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODE_PROMPTS = {
    "Email": (
        "You are a transcription cleanup assistant. Rewrite the raw speech "
        "transcript into clear, well-punctuated, professional email prose. "
        "Remove filler words and false starts, fix grammar, use complete "
        "sentences. Do not add information that was not said. Output ONLY "
        "the cleaned text."
    ),
    "Chat / Slack": (
        "You are a transcription cleanup assistant. Rewrite the raw speech "
        "transcript into clear, casual chat-message text. Remove filler "
        "words and false starts, fix grammar, but keep it brief and "
        "conversational -- do not over-formalize. Do not add information "
        "that was not said. Output ONLY the cleaned text."
    ),
    "Notes": (
        "You are a transcription cleanup assistant. Rewrite the raw speech "
        "transcript into clear, concise note form. Remove filler words and "
        "false starts, fix grammar, tighten wordy phrasing. Do not add "
        "information that was not said. Output ONLY the cleaned text."
    ),
    "Plain cleanup": (
        "You are a transcription cleanup assistant. Rewrite the raw speech "
        "transcript into clear, well-punctuated text. Remove filler words "
        "and false starts, fix grammar. Do not add information that was "
        "not said. Output ONLY the cleaned text."
    ),
}

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------

print(f"Loading ASR model from {WHISPER_REPO_ID} ...")
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_REPO_ID)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_REPO_ID,
    torch_dtype=DTYPE,
).to(DEVICE)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=DTYPE,
    device=DEVICE,
)

print(f"Loading cleanup LLM from {LLM_REPO_ID} ...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_REPO_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_REPO_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)
if DEVICE == "cpu":
    llm_model = llm_model.to(DEVICE)

print("Models loaded.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def transcribe(audio_path):
    if audio_path is None:
        return "", 0.0
    start = time.time()
    result = asr_pipe(audio_path)
    elapsed = time.time() - start
    return result["text"].strip(), elapsed


def clean_up(raw_text, mode):
    if not raw_text.strip():
        return "", 0.0
    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["Plain cleanup"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_text},
    ]
    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=llm_tokenizer.pad_token_id or llm_tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    cleaned = llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return cleaned, elapsed


def run_pipeline(audio_path, mode):
    """
    Drives the animated flow stages too: returns stage CSS classes so the
    front-end can light up Mic -> ASR -> Cleanup -> Done as it progresses.
    Gradio can't animate mid-function easily without a generator, so we
    use a generator here and yield partial UI state after each stage.
    """
    if audio_path is None:
        yield "", "", "Record or upload audio first.", _stage_html("idle")
        return

    yield "", "", "Transcribing…", _stage_html("asr")
    raw_text, asr_seconds = transcribe(audio_path)
    if not raw_text:
        yield "", "", "Couldn't transcribe that clip — try again.", _stage_html("idle")
        return

    yield raw_text, "", "Cleaning up…", _stage_html("cleanup")
    cleaned_text, llm_seconds = clean_up(raw_text, mode)

    total = asr_seconds + llm_seconds
    stats = (
        f"ASR: {asr_seconds * 1000:.0f} ms  |  "
        f"Cleanup: {llm_seconds * 1000:.0f} ms  |  "
        f"Total: {total * 1000:.0f} ms"
    )
    yield raw_text, cleaned_text, stats, _stage_html("done")


def _stage_html(active):
    """Renders the 4-node flow diagram, highlighting the active stage."""
    stages = [
        ("mic", "🎙️", "Mic"),
        ("asr", "📝", "ASR"),
        ("cleanup", "✨", "Cleanup"),
        ("done", "✅", "Output"),
    ]
    order = ["idle", "mic", "asr", "cleanup", "done"]
    active_idx = order.index(active) if active in order else 0

    nodes = []
    for i, (key, icon, label) in enumerate(stages):
        state = ""
        if active == "idle":
            state = ""
        elif order.index(key) < active_idx:
            state = "flow-done"
        elif key == active:
            state = "flow-active"
        nodes.append(
            f'<div class="flow-node {state}">'
            f'<div class="flow-icon">{icon}</div>'
            f'<div class="flow-label">{label}</div>'
            f'</div>'
        )
        if i < len(stages) - 1:
            arrow_state = "flow-done" if order.index(key) < active_idx else ""
            nodes.append(f'<div class="flow-arrow {arrow_state}">➜</div>')

    return f'<div class="flow-row">{"".join(nodes)}</div>'


# ---------------------------------------------------------------------------
# Custom CSS — flow look & feel + talk key styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    background: radial-gradient(circle at top, #131a2b 0%, #0b0f1a 60%, #05070c 100%) !important;
}

#title-md h1 {
    background: linear-gradient(90deg, #7dd3fc, #c4b5fd, #f0abfc);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: 800;
    letter-spacing: -0.5px;
}
#title-md p { color: #9aa4b8 !important; }

/* --- Flow diagram --- */
.flow-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 18px 8px;
    flex-wrap: wrap;
}
.flow-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 84px;
    height: 84px;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1.5px solid rgba(255,255,255,0.08);
    transition: all 0.35s ease;
}
.flow-icon { font-size: 24px; line-height: 1; }
.flow-label {
    margin-top: 6px;
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #8a93a6;
}
.flow-arrow {
    font-size: 20px;
    color: #3a4256;
    transition: color 0.35s ease;
}
.flow-arrow.flow-done { color: #7dd3fc; }

.flow-node.flow-active {
    border-color: #7dd3fc;
    background: rgba(125, 211, 252, 0.12);
    box-shadow: 0 0 0 4px rgba(125,211,252,0.08), 0 0 24px rgba(125,211,252,0.25);
    transform: scale(1.08);
    animation: pulse 1.1s ease-in-out infinite;
}
.flow-node.flow-active .flow-label { color: #7dd3fc; }

.flow-node.flow-done {
    border-color: rgba(125, 211, 252, 0.5);
    background: rgba(125, 211, 252, 0.06);
}
.flow-node.flow-done .flow-label { color: #7dd3fc; }

@keyframes pulse {
    0%   { box-shadow: 0 0 0 4px rgba(125,211,252,0.08), 0 0 18px rgba(125,211,252,0.2); }
    50%  { box-shadow: 0 0 0 8px rgba(125,211,252,0.14), 0 0 32px rgba(125,211,252,0.4); }
    100% { box-shadow: 0 0 0 4px rgba(125,211,252,0.08), 0 0 18px rgba(125,211,252,0.2); }
}

/* --- Talk key --- */
#talk-key-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 14px 0 4px 0;
}
#talk-key {
    position: relative;
    width: 96px;
    height: 96px;
    border-radius: 50%;
    background: linear-gradient(145deg, #1b2338, #10151f);
    border: 2px solid #2c3550;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    cursor: pointer;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
    transition: transform 0.15s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
#talk-key:hover { transform: translateY(-2px); border-color: #4a5580; }
#talk-key.recording {
    border-color: #f87171;
    background: linear-gradient(145deg, #3a1620, #1f0d12);
    box-shadow: 0 0 0 8px rgba(248,113,113,0.12), 0 0 30px rgba(248,113,113,0.35);
    animation: recPulse 1s ease-in-out infinite;
}
#talk-key.handsfree {
    border-color: #fbbf24;
    background: linear-gradient(145deg, #3a2c10, #1f1a0a);
    box-shadow: 0 0 0 8px rgba(251,191,36,0.12), 0 0 30px rgba(251,191,36,0.35);
}
@keyframes recPulse {
    0%   { box-shadow: 0 0 0 6px rgba(248,113,113,0.10), 0 0 22px rgba(248,113,113,0.3); }
    50%  { box-shadow: 0 0 0 12px rgba(248,113,113,0.18), 0 0 36px rgba(248,113,113,0.5); }
    100% { box-shadow: 0 0 0 6px rgba(248,113,113,0.10), 0 0 22px rgba(248,113,113,0.3); }
}
#talk-key-hint {
    font-size: 12px;
    color: #7b859c;
    text-align: center;
    max-width: 260px;
    line-height: 1.5;
}
#talk-key-status {
    font-size: 13px;
    font-weight: 600;
    color: #cbd5e1;
    min-height: 18px;
}

#stats-md { text-align: center; color: #7dd3fc !important; font-size: 13px; }
"""

# ---------------------------------------------------------------------------
# JS: press/hold = push-to-talk, double-click = hands-free toggle.
# Uses MediaRecorder directly and drops the recorded blob into a hidden
# gr.Audio component (as a base64 data URL) which Gradio decodes into the
# same audio_input the rest of the pipeline already consumes.
# ---------------------------------------------------------------------------

TALK_KEY_JS = """
async () => {
    const key = document.getElementById("talk-key");
    const status = document.getElementById("talk-key-status");
    const audioInput = document.querySelector("#hidden-audio-target input[type='file'], #hidden-audio-target textarea");

    // We drive the hidden gr.Audio (type='filepath') via its base64 sibling:
    // simplest robust route is a hidden gr.Textbox holding a base64 wav
    // string, decoded by Python. See `decode_and_run` bound below.
    const hiddenBox = document.querySelector("#hidden-b64 textarea");

    let mediaRecorder = null;
    let chunks = [];
    let mode = "idle"; // idle | hold | handsfree
    let clickTimer = null;
    let stream = null;

    function setStatus(text) { if (status) status.innerText = text; }
    function setKeyState(cls) {
        key.classList.remove("recording", "handsfree");
        if (cls) key.classList.add(cls);
    }

    async function startRecording() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
            setStatus("Microphone permission denied.");
            return false;
        }
        chunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
        mediaRecorder.start();
        return true;
    }

    function stopRecording() {
        return new Promise((resolve) => {
            if (!mediaRecorder || mediaRecorder.state === "inactive") { resolve(null); return; }
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: "audio/webm" });
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result); // data URL
                reader.readAsDataURL(blob);
                stream.getTracks().forEach(t => t.stop());
            };
            mediaRecorder.stop();
        });
    }

    async function finishAndSend() {
        setStatus("Processing…");
        setKeyState(null);
        const dataUrl = await stopRecording();
        mode = "idle";
        if (!dataUrl) { setStatus("No audio captured."); return; }
        if (hiddenBox) {
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            nativeSetter.call(hiddenBox, dataUrl);
            hiddenBox.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    key.addEventListener("mousedown", async (e) => {
        if (mode === "handsfree") return; // let click handler manage handsfree stop
        mode = "hold";
        setKeyState("recording");
        setStatus("Listening… (release to send)");
        await startRecording();
    });

    key.addEventListener("mouseup", async () => {
        if (mode !== "hold") return;
        await finishAndSend();
    });

    key.addEventListener("mouseleave", async () => {
        if (mode === "hold" && mediaRecorder && mediaRecorder.state === "recording") {
            await finishAndSend();
        }
    });

    key.addEventListener("dblclick", async (e) => {
        e.preventDefault();
        if (mode === "hold") return;
        if (mode === "handsfree") {
            await finishAndSend();
            return;
        }
        mode = "handsfree";
        setKeyState("handsfree");
        setStatus("Hands-free… (click once to stop)");
        await startRecording();
    });

    key.addEventListener("click", async (e) => {
        if (mode === "handsfree" && e.detail === 1) {
            // small delay to distinguish from dblclick which also fires two clicks
            setTimeout(async () => {
                if (mode === "handsfree") {
                    await finishAndSend();
                }
            }, 250);
        }
    });

    // Spacebar: hold = push-to-talk. Double-tap space quickly = hands-free.
    let lastSpaceTime = 0;
    let spaceHeld = false;
    document.addEventListener("keydown", async (e) => {
        if (e.code !== "Space") return;
        const tag = document.activeElement.tagName;
        if (tag === "TEXTAREA" || tag === "INPUT") return;
        e.preventDefault();
        if (spaceHeld) return;
        const now = Date.now();
        if (now - lastSpaceTime < 350 && mode === "idle") {
            mode = "handsfree";
            setKeyState("handsfree");
            setStatus("Hands-free… (press space to stop)");
            await startRecording();
            spaceHeld = true;
            lastSpaceTime = 0;
            return;
        }
        lastSpaceTime = now;
        if (mode === "handsfree") {
            await finishAndSend();
            spaceHeld = true;
            return;
        }
        if (mode === "idle") {
            mode = "hold";
            setKeyState("recording");
            setStatus("Listening… (release space to send)");
            await startRecording();
        }
        spaceHeld = true;
    });
    document.addEventListener("keyup", async (e) => {
        if (e.code !== "Space") return;
        spaceHeld = false;
        if (mode === "hold") {
            await finishAndSend();
        }
    });
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Voice → Clean Text", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# 🎙️ Voice → Clean Text\n"
        "Press-and-hold the key to talk. Double-click (or double-tap space) "
        "for hands-free mode.",
        elem_id="title-md",
    )

    flow_display = gr.HTML(_stage_html("idle"))

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            with gr.Group(elem_id="talk-key-wrap"):
                gr.HTML('<div id="talk-key">🎙️</div>')
                gr.HTML('<div id="talk-key-status">Hold to talk · Double-click for hands-free</div>')
                gr.HTML(
                    '<div id="talk-key-hint">'
                    'You can also just hold the <b>spacebar</b>. Double-tap it fast '
                    'for hands-free mode.</div>'
                )

            mode = gr.Radio(
                choices=list(MODE_PROMPTS.keys()),
                value="Plain cleanup",
                label="Cleanup style",
            )

            with gr.Accordion("Or upload / record with the classic widget", open=False):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Speak or upload audio",
                )
                run_btn = gr.Button("Transcribe & Clean Up", variant="primary")

        with gr.Column(scale=2):
            raw_output = gr.Textbox(label="Raw transcript (verbatim)", lines=4)
            clean_output = gr.Textbox(label="Cleaned text", lines=4)
            stats_output = gr.Markdown(elem_id="stats-md")

    # Hidden bridge: JS writes a base64 data URL of the recorded clip here;
    # Python decodes it to a temp wav file and feeds the normal pipeline.
    hidden_b64 = gr.Textbox(visible=False, elem_id="hidden-b64")

    def decode_and_run(data_url, mode_value):
        import base64
        import tempfile

        if not data_url or "," not in data_url:
            yield "", "", "No audio captured.", _stage_html("idle")
            return
        header, encoded = data_url.split(",", 1)
        raw_bytes = base64.b64decode(encoded)
        suffix = ".webm" if "webm" in header else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw_bytes)
            tmp_path = f.name
        yield from run_pipeline(tmp_path, mode_value)

    hidden_b64.change(
        fn=decode_and_run,
        inputs=[hidden_b64, mode],
        outputs=[raw_output, clean_output, stats_output, flow_display],
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, mode],
        outputs=[raw_output, clean_output, stats_output, flow_display],
    )
    audio_input.stop_recording(
        fn=run_pipeline,
        inputs=[audio_input, mode],
        outputs=[raw_output, clean_output, stats_output, flow_display],
    )

    demo.load(fn=None, js=TALK_KEY_JS)

if __name__ == "__main__":
    demo.queue().launch()