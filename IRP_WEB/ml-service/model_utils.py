import os
import re
import base64
import random
from io import BytesIO
from typing import Optional, Dict, Any
from collections import deque

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH")
FER_MODEL_PATH = os.getenv("FER_MODEL_PATH")
GEN_MODEL_PATH = os.getenv("GEN_MODEL_PATH")

TEXT_LABEL_ORDER = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral"
]

FER_LABEL_ORDER = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "sadness",
    5: "surprise",
    6: "neutral"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7

RESPONSE_MODE = "both"   # template / generator / both

# Session memory: each session keeps short-term history
SESSION_MEMORY: Dict[str, Dict[str, deque]] = {}

# Crisis support details used here are the Sri Lankan NIMH 1926 helpline and NIMH main line. 
CRISIS_LINES = {
    "helpline": "1926",
    "nimh_main": "+94 11 257 8234–7"
}

CRISIS_PATTERNS = [
    r"\bsuicidal\b",
    r"\bsuicidal thoughts\b",
    r"\bkill myself\b",
    r"\bwant to die\b",
    r"\bend my life\b",
    r"\bdon'?t want to live\b",
    r"\bcannot go on\b",
    r"\bcan'?t go on\b",
    r"\bhurt myself\b",
    r"\bself harm\b",
    r"\bno reason to live\b",
    r"\bi feel hopeless\b",
    r"\bi am hopeless\b",
    r"\bi can't bear this\b",
    r"\bi cannot bear this\b",
    r"\bi want to disappear\b",
    r"\bi want to hurt myself\b"
]

# =========================
# LOAD MODELS ON STARTUP
# =========================

print(f"Loading models on {DEVICE}...")

# Text model
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
text_model.to(DEVICE)
text_model.eval()

# Generator model
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=False)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_PATH)
gen_model.to(DEVICE)
gen_model.eval()

# FER model
def get_vgg19_model():
    model = models.vgg19(weights=None)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    return model

fer_model = get_vgg19_model()
state_dict = torch.load(FER_MODEL_PATH, map_location=DEVICE)
fer_model.load_state_dict(state_dict)
fer_model.to(DEVICE)
fer_model.eval()

fer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485] * 3, [0.229] * 3)
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Models loaded successfully.")

# =========================
# SESSION MEMORY
# =========================

def get_session_memory(session_id: str):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = {
            "conversation_history": deque(maxlen=5),
            "final_emotion_history": deque(maxlen=5),
            "fusion_mode_history": deque(maxlen=5),
        }
    return SESSION_MEMORY[session_id]


def update_memory(session_id: str, user_text: str, final_emotion: str, fusion_mode: str):
    mem = get_session_memory(session_id)
    mem["conversation_history"].append(user_text)
    mem["final_emotion_history"].append(final_emotion)
    mem["fusion_mode_history"].append(fusion_mode)


# =========================
# IMAGE HELPERS
# =========================

def decode_base64_image(image_data: Optional[str]):
    if not image_data:
        return None
    try:
        _, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


# =========================
# TEXT MODEL
# =========================

def predict_text_emotion(text: str):
    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = text_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = TEXT_LABEL_ORDER[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs


# =========================
# FACE MODEL
# =========================

def predict_face_emotion_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, 0.0, np.zeros(len(TEXT_LABEL_ORDER), dtype=np.float32)

    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = faces[0]
    face_crop = frame[y:y+h, x:x+w]

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray_face).convert("RGB")
    x_tensor = fer_transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = fer_model(x_tensor)
        raw_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    raw_idx = int(np.argmax(raw_probs))
    raw_conf = float(raw_probs[raw_idx])
    mapped_label = FER_LABEL_ORDER[raw_idx]

    unified_probs = np.zeros(len(TEXT_LABEL_ORDER), dtype=np.float32)
    for i, p in enumerate(raw_probs):
        mapped = FER_LABEL_ORDER[i]
        unified_idx = TEXT_LABEL_ORDER.index(mapped)
        unified_probs[unified_idx] += float(p)

    return mapped_label, raw_conf, unified_probs


# =========================
# FUSION
# =========================

def detect_emotion_conflict(text_label: str, face_label: Optional[str]):
    if face_label is None:
        return False
    return text_label != face_label


def fuse_emotions(text_probs, face_probs, text_conf, face_conf,
                  text_priority_threshold=0.70,
                  face_threshold=0.55):

    if text_conf >= text_priority_threshold:
        final_probs = text_probs
        mode = "text-priority"

    elif face_conf < face_threshold:
        final_probs = text_probs
        mode = "text-only fallback"

    else:
        total_conf = text_conf + face_conf
        alpha_text = text_conf / total_conf
        alpha_face = face_conf / total_conf
        final_probs = alpha_text * text_probs + alpha_face * face_probs
        mode = "weighted fusion"

    final_idx = int(np.argmax(final_probs))
    final_label = TEXT_LABEL_ORDER[final_idx]
    final_conf = float(final_probs[final_idx])

    return final_label, final_conf, final_probs, mode


# =========================
# MOOD TRACKING
# =========================

def detect_mood_trend(session_id: str):
    mem = get_session_memory(session_id)
    hist = mem["final_emotion_history"]

    if len(hist) < 3:
        return "insufficient-history"

    last_three = list(hist)[-3:]
    negative = {"sadness", "anger", "fear", "disgust"}

    if len(set(last_three)) == 1 and last_three[0] in negative:
        return "persistent-negative"

    if len(set(last_three)) == 3:
        return "emotional-fluctuation"

    if last_three[-1] == "joy":
        return "possible-improvement"

    return "stable"


def build_memory_hint(session_id: str):
    mem = get_session_memory(session_id)
    hist = mem["final_emotion_history"]

    if len(hist) < 2:
        return None

    last_two = list(hist)[-2:]
    last_three = list(hist)[-3:]

    negative_emotions = {"sadness", "anger", "fear", "disgust"}

    if all(e in negative_emotions for e in last_two):
        return random.choice([
            "It seems like these feelings have been staying with you for a while.",
            "It sounds like this has been weighing on you for some time.",
            "You might have been dealing with these feelings for longer than just now."
        ])

    if len(last_three) == 3 and len(set(last_three)) >= 3:
        return random.choice([
            "It seems like your feelings have been changing quite a bit recently.",
            "It sounds like your emotions have been shifting over time.",
            "You might be going through a mix of different feelings lately."
        ])

    if last_two[-1] == "joy":
        return random.choice([
            "It sounds like things may be starting to feel a little better.",
            "There seems to be a small positive shift in how you're feeling.",
            "It feels like there might be a bit of improvement recently."
        ])

    return None


# =========================
# RESPONSE LOGIC
# =========================

def detect_crisis(user_text: str):
    t = user_text.lower().strip()

    for pattern in CRISIS_PATTERNS:
        if re.search(pattern, t):
            return True

    # EXTRA SAFETY: combination logic
    if ("stress" in t or "pain" in t or "overwhelmed" in t) and (
        "can't" in t or "cannot" in t or "bear" in t
    ):
        return True

    return False


def generate_safe_crisis_response():
    responses = [
        (
            f"I’m really sorry you’re feeling this overwhelmed right now. "
            f"Please contact Sri Lanka’s National Mental Health Helpline on {CRISIS_LINES['helpline']} "
            f"or reach the National Institute of Mental Health on {CRISIS_LINES['nimh_main']} as soon as possible. "
            f"If you feel unsafe right now, go to the nearest hospital or get immediate help from someone physically near you."
        ),
        (
            f"It sounds like you may be in serious distress, and your safety matters most right now. "
            f"Please call {CRISIS_LINES['helpline']} for urgent mental health support in Sri Lanka, "
            f"or contact the National Institute of Mental Health on {CRISIS_LINES['nimh_main']}. "
            f"Please also reach out to a trusted family member, friend, or nearby emergency service immediately."
        ),
        (
            f"I’m very sorry that things feel this unbearable right now. "
            f"Please seek immediate support by calling {CRISIS_LINES['helpline']}, which is Sri Lanka’s National Mental Health Helpline, "
            f"or contact the National Institute of Mental Health on {CRISIS_LINES['nimh_main']}. "
            f"If you might act on these thoughts, go to the nearest hospital or ask someone with you to help right now."
        )
    ]
    return random.choice(responses)


def get_emotion_template(emotion):
    templates = {
        "sadness": {
            "openers": [
                "I’m really sorry you’re carrying this right now.",
                "That sounds deeply difficult to sit with.",
                "It makes sense that this feels heavy for you."
            ],
            "supports": [
                "It can take time to process something painful like this.",
                "Moments like this can feel overwhelming, especially when everything hits at once.",
                "It’s not easy to carry something like this, especially when it affects so much at once."
            ],
            "questions": [
                "What feels heaviest for you at the moment?",
                "What has been the hardest part of this for you?",
                "Would you like to share a little more about what happened?"
            ]
        },
        "anger": {
            "openers": [
                "That sounds really frustrating.",
                "I can understand why you feel upset about this.",
                "It makes sense that this brought up a lot of anger."
            ],
            "supports": [
                "When something feels unfair, those reactions can be intense.",
                "It may help to slow the moment down before deciding what to do next.",
                "Your reaction matters, and it is worth understanding what triggered it."
            ],
            "questions": [
                "What part of this upset you the most?",
                "What do you think triggered this reaction most strongly?",
                "What happened just before you felt this way?"
            ]
        },
        "fear": {
            "openers": [
                "That sounds really scary.",
                "I can see why this would make you feel anxious.",
                "It makes sense that you’re feeling worried right now."
            ],
            "supports": [
                "When things feel uncertain, the mind can rush ahead quickly.",
                "You do not have to solve everything at once right now.",
                "It may help to focus on the next small step rather than the whole situation."
            ],
            "questions": [
                "What feels most uncertain to you right now?",
                "What part of this is worrying you the most?",
                "What would help you feel a little safer in this moment?"
            ]
        },
        "joy": {
            "openers": [
                "That sounds wonderful.",
                "I’m really glad something good happened for you.",
                "It’s lovely to hear that you’re feeling this way."
            ],
            "supports": [
                "Moments like this can be really meaningful.",
                "It’s worth pausing to appreciate what made this feel special.",
                "Holding on to positive moments like this can be really valuable."
            ],
            "questions": [
                "What made this moment feel especially meaningful?",
                "What part of it made you happiest?",
                "How would you like to hold on to this feeling?"
            ]
        },
        "surprise": {
            "openers": [
                "That sounds really unexpected.",
                "I can see why that caught you off guard.",
                "That must have been surprising to process."
            ],
            "supports": [
                "Unexpected moments can take time to make sense of.",
                "Sometimes the first reaction is just trying to understand what happened.",
                "It’s okay if you’re still processing it."
            ],
            "questions": [
                "What stood out to you most about it?",
                "How are you making sense of it right now?",
                "What was your first reaction when it happened?"
            ]
        },
        "neutral": {
            "openers": [
                "Thanks for sharing that.",
                "I’m here with you.",
                "I appreciate you telling me that."
            ],
            "supports": [
                "Even ordinary moments can still carry a lot underneath them.",
                "Sometimes it helps to pause and notice how the day actually felt.",
                "You can take your time with this."
            ],
            "questions": [
                "How did that day feel for you overall?",
                "Was there any part of it that stayed with you?",
                "Would you like to say a bit more about that?"
            ]
        },
        "disgust": {
            "openers": [
                "That sounds really uncomfortable.",
                "I can understand why that would feel upsetting.",
                "That reaction makes sense."
            ],
            "supports": [
                "Some situations can leave a strong emotional reaction behind.",
                "It can help to name exactly what felt wrong about it.",
                "You do not need to dismiss that reaction."
            ],
            "questions": [
                "What exactly felt most wrong about it?",
                "What part stayed with you most strongly?",
                "Would you like to talk more about what triggered that reaction?"
            ]
        }
    }
    return templates.get(emotion, templates["neutral"])


def get_response_strategy(emotion, mood_trend):
    if mood_trend == "persistent-negative":
        return "acknowledge persistence, validate distress, gently invite reflection on what has been repeating"
    if mood_trend == "emotional-fluctuation":
        return "acknowledge changing emotions, stay calm, offer grounding and simple reflection"
    if emotion == "sadness":
        return "validate sadness, normalize emotional pain, gently explore the heaviest part"
    elif emotion == "anger":
        return "validate frustration, reduce escalation, encourage reflection on trigger"
    elif emotion == "fear":
        return "reassure without dismissing, reduce overwhelm, focus on immediate concern"
    elif emotion == "joy":
        return "affirm positive emotion, reinforce meaning, encourage reflection on what helped"
    elif emotion == "surprise":
        return "acknowledge unexpectedness, help user make sense of event"
    elif emotion == "neutral":
        return "stay natural, open, and curious"
    elif emotion == "disgust":
        return "validate discomfort, gently explore source of reaction"
    return "respond supportively and ask one gentle follow-up question"


def trim_to_word_range(text, min_words=30, max_words=60):
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        text = " ".join(words).rstrip(",;:") + "."
    return text


def compose_normal_response(session_id: str, user_text: str, final_emotion: str, mood_trend: str, strategy: str):
    bank = get_emotion_template(final_emotion)

    opener = random.choice(bank["openers"])
    support = random.choice(bank["supports"])

    if mood_trend == "persistent-negative":
        question = random.choice([
            "What feels like it has been weighing on you the most lately?",
            "Do you feel like this has been building up over time?",
            "Would you like to share what has been hardest to deal with recently?"
        ])
    else:
        question = random.choice(bank["questions"])

    memory_hint = build_memory_hint(session_id)

    include_memory = random.choice([True, False])
    include_cbt = random.choice([True, True, False])

    cbt_lines = {
        "sadness": [
            "It might help to focus on just one small part of this rather than everything at once.",
            "Sometimes putting the feeling into words can make it a little easier to carry."
        ],
        "anger": [
            "Taking a step back for a moment can sometimes help you see things more clearly.",
            "It might help to separate what happened from how it made you feel."
        ],
        "fear": [
            "It can help to bring your attention back to what is happening right now instead of everything at once.",
            "Focusing on one small next step can sometimes ease the pressure."
        ],
        "joy": [
            "Moments like this can be really meaningful when you pause and take them in.",
            "It’s nice to recognize what helped create this feeling."
        ],
        "surprise": [
            "It can take a little time to fully process something unexpected like this.",
            "Sometimes the first step is just making sense of what happened."
        ],
        "neutral": [
            "Sometimes even simple moments can carry more meaning than we expect.",
            "It might be worth noticing how the day actually felt for you."
        ],
        "disgust": [
            "Strong reactions like this can tell you something important about what matters to you.",
            "It can help to understand exactly what caused that reaction."
        ]
    }

    sentences = [opener]

    if mood_trend == "persistent-negative":
        sentences.insert(1, "It sounds like this feeling has been staying with you for some time, and that can be really exhausting.")

    if memory_hint and include_memory:
        sentences.append(memory_hint)

    sentences.append(support)

    if include_cbt:
        sentences.append(random.choice(cbt_lines.get(final_emotion, cbt_lines["neutral"])))

    sentences.append(question)

    core_sentences = sentences[:-1]
    question_sentence = sentences[-1]

    num_core = random.choice([2, 3])
    selected_core = core_sentences[:num_core]
    selected = selected_core + [question_sentence]

    response = " ".join(selected)

    if len(response.split()) < 30:
        for s in sentences:
            if s not in selected:
                response += " " + s
                if len(response.split()) >= 30:
                    break

    response = trim_to_word_range(response, 30, 60)
    return response


def generate_model_response(user_text: str, final_emotion: str, mood_trend: str, strategy: str, session_id: str):
    mem = get_session_memory(session_id)
    history = "\n".join([
        f"Turn {i+1}: {msg} | Emotion={emo}"
        for i, (msg, emo) in enumerate(zip(mem["conversation_history"], mem["final_emotion_history"]))
    ]) or "No previous conversation history."

    prompt = f"""
You are a supportive mental health assistant.

Current emotion: {final_emotion}
Mood trend: {mood_trend}
Support strategy: {strategy}

Recent conversation history:
{history}

Current user message:
{user_text}

Write 3 to 4 sentences.
Be warm, coherent, emotionally supportive, and natural.
The final sentence should gently invite the user to continue sharing.
Do not repeat phrases.
Do not give medical advice.
"""

    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384
    ).to(DEVICE)

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True
        )

    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response


def generate_final_response(session_id: str, user_text: str, final_emotion: str, mood_trend: str, strategy: str):
    if RESPONSE_MODE == "template":
        return compose_normal_response(session_id, user_text, final_emotion, mood_trend, strategy)

    if RESPONSE_MODE == "generator":
        return generate_model_response(user_text, final_emotion, mood_trend, strategy, session_id)

    # both -> use template as primary safe final output
    # generator kept available experimentally
    return compose_normal_response(session_id, user_text, final_emotion, mood_trend, strategy)


# =========================
# MAIN PREDICTION FUNCTION
# =========================

def multimodal_predict(text: str, image_data: Optional[str] = None,
                       camera_enabled: bool = False, session_id: str = "default") -> Dict[str, Any]:

    text = (text or "").strip()
    if not text:
        return {
            "text_emotion": "neutral",
            "text_conf": 0.0,
            "face_emotion": None,
            "face_conf": 0.0,
            "final_emotion": "neutral",
            "final_conf": 0.0,
            "fusion_mode": "text-only fallback",
            "conflict": False,
            "mood_trend": "insufficient-history",
            "strategy": "stay natural, open, and curious",
            "safety_mode": "normal-mode",
            "response": "Please tell me a little about how you're feeling."
        }

    crisis_mode = detect_crisis(text)

    text_label, text_conf, text_probs = predict_text_emotion(text)

    face_label = None
    face_conf = 0.0
    face_probs = np.zeros(len(TEXT_LABEL_ORDER), dtype=np.float32)

    if camera_enabled and image_data:
        frame = decode_base64_image(image_data)
        if frame is not None:
            face_label, face_conf, face_probs = predict_face_emotion_from_frame(frame)

    conflict = detect_emotion_conflict(text_label, face_label)

    final_label, final_conf, final_probs, fusion_mode = fuse_emotions(
        text_probs=text_probs,
        face_probs=face_probs,
        text_conf=text_conf,
        face_conf=face_conf
    )

    mood_trend = detect_mood_trend(session_id)
    strategy = get_response_strategy(final_label, mood_trend)

    if crisis_mode:
        response = generate_safe_crisis_response()
        safety_mode = "crisis-safe-mode"
    else:
        response = generate_final_response(
            session_id=session_id,
            user_text=text,
            final_emotion=final_label,
            mood_trend=mood_trend,
            strategy=strategy
        )
        safety_mode = "normal-mode"

    update_memory(session_id, text, final_label, fusion_mode)

    return {
        "text_emotion": text_label,
        "text_conf": float(text_conf),
        "face_emotion": face_label,
        "face_conf": float(face_conf),
        "final_emotion": final_label,
        "final_conf": float(final_conf),
        "fusion_mode": fusion_mode,
        "conflict": bool(conflict),
        "mood_trend": mood_trend,
        "strategy": strategy,
        "safety_mode": safety_mode,
        "response": response
    }