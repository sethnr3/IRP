import { useEffect, useRef, useState } from "react";
import {
  Camera,
  CameraOff,
  Send,
  AlertTriangle,
  ShieldCheck,
  Wifi,
  WifiOff,
  Bot,
  User,
  Sparkles,
} from "lucide-react";

const API_URL = "http://localhost:5000/api/multimodal-chat";

const emotionColors = {
  anger: "bg-red-100 text-red-800 border-red-200",
  disgust: "bg-lime-100 text-lime-800 border-lime-200",
  fear: "bg-purple-100 text-purple-800 border-purple-200",
  joy: "bg-amber-100 text-amber-800 border-amber-200",
  sadness: "bg-blue-100 text-blue-800 border-blue-200",
  surprise: "bg-orange-100 text-orange-800 border-orange-200",
  neutral: "bg-slate-100 text-slate-700 border-slate-200",
};

const starterMessages = [
  "Hi there 👋 I'm here to support you. How are you feeling today?",
  "Hello 😊 You can talk to me about anything on your mind.",
  "Hey 💙 I'm here to listen. What's going on?",
  "Welcome 🌿 Tell me how you're feeling, and I'll do my best to help.",
  "Hi! Let's begin gently — how has your day been so far?",
];

function labelize(value) {
  if (!value) return "Not available";
  return value
    .split("-")
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}

function formatConfidence(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "-";
  return `${(n * 100).toFixed(0)}%`;
}

function getEmotionEmoji(emotion) {
  const map = {
    anger: "😠",
    disgust: "🤢",
    fear: "😨",
    joy: "😊",
    sadness: "😢",
    surprise: "😲",
    neutral: "😐",
  };
  return map[emotion] || "💭";
}

function StatusBadge({ online }) {
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${
        online
          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
          : "border-red-200 bg-red-50 text-red-700"
      }`}
    >
      {online ? <Wifi size={14} /> : <WifiOff size={14} />}
      {online ? "Backend Connected" : "Backend Offline"}
    </span>
  );
}

function SoftBadge({ children }) {
  return (
    <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-700">
      {children}
    </span>
  );
}

function MetaItem({ label, value, emotion }) {
  const emotionClass =
    emotionColors[emotion] || "bg-slate-100 text-slate-700 border-slate-200";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <span
        className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${
          emotion ? emotionClass : "bg-slate-100 text-slate-700 border-slate-200"
        }`}
      >
        {emotion && <span className="text-sm">{getEmotionEmoji(emotion)}</span>}
        {value}
      </span>
    </div>
  );
}

function TypingBubble() {
  return (
    <div
      className="flex justify-start"
      style={{ animation: "fadeInUp 0.35s ease" }}
    >
      <div className="max-w-[90%] rounded-3xl border border-slate-200 bg-white px-4 py-4 text-slate-900 shadow-sm sm:max-w-[85%]">
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-slate-500">
          <Bot size={14} />
          Assistant
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-600">Assistant is typing</span>
          <div className="flex items-center gap-1">
            <span
              className="h-2 w-2 rounded-full bg-slate-400"
              style={{ animation: "typingBounce 1.2s infinite 0s" }}
            />
            <span
              className="h-2 w-2 rounded-full bg-slate-400"
              style={{ animation: "typingBounce 1.2s infinite 0.2s" }}
            />
            <span
              className="h-2 w-2 rounded-full bg-slate-400"
              style={{ animation: "typingBounce 1.2s infinite 0.4s" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const meta = message.meta;

  return (
    <div
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
      style={{ animation: "fadeInUp 0.35s ease" }}
    >
      <div
        className={`max-w-[90%] rounded-3xl px-4 py-4 shadow-sm sm:max-w-[85%] ${
          isUser
            ? "bg-slate-900 text-white"
            : "border border-slate-200 bg-white text-slate-900"
        }`}
      >
        <div
          className={`mb-2 flex items-center gap-2 text-xs font-semibold ${
            isUser ? "text-slate-300" : "text-slate-500"
          }`}
        >
          {isUser ? <User size={14} /> : <Bot size={14} />}
          {isUser ? "You" : "Assistant"}
        </div>

        <div className="whitespace-pre-wrap text-sm leading-7 sm:text-[15px]">
          {message.text}
        </div>

        {!isUser && meta && (
          <div className="mt-4 grid grid-cols-1 gap-3 rounded-2xl bg-slate-50 p-3 sm:grid-cols-2">
            <MetaItem
              label="Text Emotion"
              value={`${labelize(meta.text_emotion)} (${formatConfidence(
                meta.text_conf
              )})`}
              emotion={meta.text_emotion}
            />
            <MetaItem
              label="Face Emotion"
              value={`${labelize(meta.face_emotion)} (${formatConfidence(
                meta.face_conf
              )})`}
              emotion={meta.face_emotion}
            />
            <MetaItem
              label="Final Emotion"
              value={`${labelize(meta.final_emotion)} (${formatConfidence(
                meta.final_conf
              )})`}
              emotion={meta.final_emotion}
            />
            <MetaItem label="Fusion" value={labelize(meta.fusion_mode)} />
            <MetaItem label="Mood Trend" value={labelize(meta.mood_trend)} />
            <MetaItem label="Safety" value={labelize(meta.safety_mode)} />
            <MetaItem label="Conflict" value={meta.conflict ? "Yes" : "No"} />
          </div>
        )}
      </div>
    </div>
  );
}

function ConsentModal({ onAccept, onDecline }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm">
      <div
        className="w-full max-w-2xl rounded-3xl bg-white p-6 shadow-2xl"
        style={{ animation: "fadeInScale 0.28s ease" }}
      >
        <h2 className="text-2xl font-bold text-slate-900">
          Allow camera for passive emotion capture?
        </h2>

        <p className="mt-3 text-sm leading-7 text-slate-600">
          The system can use your camera to improve emotion understanding during
          chat. A frame is captured only when you send a message. You can
          disable the camera at any time.
        </p>

        <div className="mt-5 flex items-start gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          <AlertTriangle className="mt-0.5 shrink-0" size={18} />
          <span>
            Use camera only with clear user consent. Transparency and user
            control are essential.
          </span>
        </div>

        <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-end">
          <button
            onClick={onDecline}
            className="rounded-2xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Continue without camera
          </button>
          <button
            onClick={onAccept}
            className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
          >
            Allow camera
          </button>
        </div>
      </div>
    </div>
  );
}

export default function MultimodalChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [cameraConsentOpen, setCameraConsentOpen] = useState(true);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [streamError, setStreamError] = useState("");
  const [sending, setSending] = useState(false);
  const [serverOnline, setServerOnline] = useState(true);
  const [assistantTyping, setAssistantTyping] = useState(true);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      const randomMessage =
        starterMessages[Math.floor(Math.random() * starterMessages.length)];

      setMessages([
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: randomMessage,
        },
      ]);
      setAssistantTyping(false);
    }, 1200);

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, assistantTyping]);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraReady(false);
  };

  const startCamera = async () => {
    try {
      setStreamError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraReady(true);
    } catch (error) {
      console.error(error);
      setCameraEnabled(false);
      setCameraReady(false);
      setStreamError(
        "Camera access failed. You can continue using text-only mode."
      );
    }
  };

  const handleConsent = async (accepted) => {
    setCameraConsentOpen(false);

    if (accepted) {
      setCameraEnabled(true);
      await startCamera();
    }
  };

  const handleToggleCamera = async () => {
    if (cameraEnabled) {
      setCameraEnabled(false);
      stopCamera();
    } else {
      setCameraEnabled(true);
      await startCamera();
    }
  };

  const captureFrame = () => {
    if (
      !cameraEnabled ||
      !cameraReady ||
      !videoRef.current ||
      !canvasRef.current
    ) {
      return null;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, width, height);
    return canvas.toDataURL("image/jpeg", 0.85);
  };

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed || sending || assistantTyping) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setSending(true);
    setAssistantTyping(true);

    const frame = captureFrame();

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: trimmed,
          image: frame,
          cameraEnabled: Boolean(frame),
        }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setServerOnline(true);

      setTimeout(() => {
        const assistantMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.response,
          meta: data,
        };

        setMessages((prev) => [...prev, assistantMessage]);
        setAssistantTyping(false);
        setSending(false);
      }, 900);
    } catch (error) {
      console.error(error);
      setServerOnline(false);

      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            text: "I couldn’t reach the backend just now. Please check whether your API server is running and try again.",
          },
        ]);
        setAssistantTyping(false);
        setSending(false);
      }, 900);
    }
  };

  return (
    <div className="h-screen overflow-hidden bg-slate-50 p-3 sm:p-4 lg:p-5">
      <style>
        {`
          @keyframes fadeInUp {
            from {
              opacity: 0;
              transform: translateY(12px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }

          @keyframes fadeInScale {
            from {
              opacity: 0;
              transform: scale(0.96);
            }
            to {
              opacity: 1;
              transform: scale(1);
            }
          }

          @keyframes typingBounce {
            0%, 60%, 100% {
              transform: translateY(0);
              opacity: 0.45;
            }
            30% {
              transform: translateY(-5px);
              opacity: 1;
            }
          }
        `}
      </style>

      {cameraConsentOpen && (
        <ConsentModal
          onAccept={() => handleConsent(true)}
          onDecline={() => handleConsent(false)}
        />
      )}

      <div className="mx-auto grid h-full max-w-7xl grid-cols-1 gap-4 xl:grid-cols-[1.6fr_0.9fr]">
        <section className="flex h-full min-h-0 flex-col rounded-3xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
          <div className="mb-4 flex shrink-0 flex-col gap-4 border-b border-slate-200 pb-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <div className="mb-2 inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
                <Sparkles size={14} />
                Emotion-Aware Support
              </div>

              <h1 className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
                Mind Support Assistant
              </h1>

              <p className="mt-2 max-w-3xl text-sm leading-7 text-slate-600 sm:text-base">
                Multimodal mental health support with text emotion, facial
                emotion, fusion, and safe-response handling.
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <StatusBadge online={serverOnline} />
              <SoftBadge>Consent-Based Camera</SoftBadge>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto rounded-3xl border border-slate-200 bg-slate-50 p-4">
            {messages.length === 0 && !assistantTyping && (
              <div className="rounded-3xl border border-dashed border-slate-300 bg-white p-6 text-sm leading-7 text-slate-600">
                Start by typing a message. If the camera is enabled, the system
                will capture a frame only when you press send.
              </div>
            )}

            <div className="space-y-4">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}

              {assistantTyping && <TypingBubble />}

              <div ref={bottomRef} />
            </div>
          </div>

          <div className="mt-4 shrink-0">
            <div className="flex flex-col gap-3 sm:flex-row">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Type how you're feeling..."
                className="h-12 flex-1 rounded-2xl border border-slate-300 bg-white px-4 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-slate-500 focus:ring-2 focus:ring-slate-200"
              />

              <button
                onClick={sendMessage}
                disabled={sending || assistantTyping || !input.trim()}
                className="inline-flex h-12 items-center justify-center rounded-2xl bg-slate-900 px-5 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Send size={16} className="mr-2" />
                Send
              </button>
            </div>

            <p className="mt-3 text-xs leading-6 text-slate-500">
              If the camera is enabled, one frame is captured only when you send
              a message.
            </p>
          </div>
        </section>

        <aside className="grid h-full min-h-0 gap-4">
          <div className="flex min-h-0 flex-col rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="shrink-0 text-lg font-semibold text-slate-900">
              Camera & Privacy
            </h3>

            <div className="mt-4 shrink-0 flex flex-col gap-4 rounded-2xl border border-slate-200 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="font-semibold text-slate-900">Camera access</div>
                <div className="text-sm text-slate-500">
                  Enable on-message facial emotion capture
                </div>
              </div>

              <button
                onClick={handleToggleCamera}
                className="inline-flex items-center justify-center rounded-2xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                {cameraEnabled ? <CameraOff size={16} /> : <Camera size={16} />}
                <span className="ml-2">
                  {cameraEnabled ? "Turn Off" : "Turn On"}
                </span>
              </button>
            </div>

            <div className="relative mt-4 shrink-0 aspect-[4/3] overflow-hidden rounded-3xl border border-slate-200 bg-slate-950">
              <video
                ref={videoRef}
                className="absolute inset-0 h-full w-full object-contain"
                muted
                playsInline
                autoPlay
              />

              {!cameraEnabled && (
                <div className="absolute inset-0 z-10 flex items-center justify-center text-sm font-medium text-white/80">
                  Camera disabled
                </div>
              )}

              {cameraEnabled && !cameraReady && (
                <div className="absolute inset-0 z-10 flex items-center justify-center text-sm font-medium text-white/80">
                  Starting camera...
                </div>
              )}
            </div>

            <canvas ref={canvasRef} className="hidden" />

            {streamError && (
              <div className="mt-4 shrink-0 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-800">
                {streamError}
              </div>
            )}

            <div className="mt-4 rounded-2xl bg-slate-50 p-4 text-sm leading-7 text-slate-600">
              The user consents once. After that, the system captures a frame
              only when a message is sent, processes it in real time, and
              allows the user to disable the camera anytime.
            </div>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">System Notes</h3>

            <div className="mt-4 space-y-3">
              <div className="rounded-2xl bg-slate-50 p-4 text-sm leading-7 text-slate-600">
                <strong className="text-slate-800">Fusion policy:</strong>{" "}
                Text-priority confidence-gated late fusion with fallback when
                camera is disabled, unavailable, or low-confidence.
              </div>

              <div className="rounded-2xl bg-slate-50 p-4 text-sm leading-7 text-slate-600">
                <strong className="text-slate-800">Safety policy:</strong>{" "}
                Crisis-safe mode can override normal response generation when
                high-risk language is detected.
              </div>

              

              <div className="flex items-start gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm leading-7 text-emerald-800">
                <ShieldCheck className="mt-0.5 shrink-0" size={18} />
                <span>
                  Consent, transparency, and user control are preserved in the
                  camera workflow.
                </span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}