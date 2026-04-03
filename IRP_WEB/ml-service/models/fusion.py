def fuse_emotions(text_emotion, text_conf, face_emotion, face_conf):

    if face_emotion == "No face detected":
        return text_emotion, text_conf, "text-only fallback", False

    conflict = text_emotion != face_emotion

    # Text priority fusion (your improved logic)
    if text_conf >= 0.75:
        return text_emotion, text_conf, "text-priority", conflict

    # Weighted fusion
    final_conf = (text_conf + face_conf) / 2

    if text_conf >= face_conf:
        return text_emotion, final_conf, "weighted fusion", conflict
    else:
        return face_emotion, final_conf, "weighted fusion", conflict