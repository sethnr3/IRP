import random

def generate_response(emotion):

    responses = {
        "sadness": [
            "I’m really sorry you're feeling this way. It sounds like something has been weighing on you, and it’s okay to feel that.",
            "That sounds difficult to go through. You don’t have to handle everything at once, and taking it step by step can help.",
            "It seems like you're carrying a lot right now. Sometimes talking about what hurts the most can make things a little lighter."
        ],
        "anger": [
            "I can sense your frustration. It might help to pause for a moment and reflect on what triggered this feeling.",
            "That sounds upsetting. Taking a breath and giving yourself space before reacting can sometimes help regain control."
        ],
        "fear": [
            "That sounds really overwhelming. You're not alone in feeling this way, and it's okay to take things slowly.",
            "It seems like something is worrying you. Try focusing on what you can control right now."
        ],
        "joy": [
            "That’s great to hear! It’s nice to see you feeling positive.",
            "You seem to be in a good place right now. Keep embracing those moments."
        ],
        "neutral": [
            "Thanks for sharing. How has your day been overall?",
            "I’m here with you. Feel free to share anything on your mind."
        ]
    }

    return random.choice(responses.get(emotion, responses["neutral"]))