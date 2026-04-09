# src/postprocessor.py

import random


# ── Hashtag banks ─────────────────────────────────────────────────────────────
HASHTAGS = {
    "people" : ["#vibes", "#lifestyle", "#mood", "#authentic", "#reallife",
                "#selfie", "#portrait", "#peopleofinsta", "#humanstory", "#moments"],
    "nature" : ["#nature", "#outdoors", "#explore", "#wanderlust", "#earthpix",
                "#naturephotography", "#wildlife", "#green", "#sky", "#beautiful"],
    "food"   : ["#foodie", "#yummy", "#instafood", "#foodporn", "#delicious",
                "#homemade", "#eating", "#foodlover", "#tasty", "#hungry"],
    "sports" : ["#fitness", "#workout", "#athlete", "#gameday", "#motivation",
                "#sport", "#gym", "#training", "#fitlife", "#strong"],
    "travel" : ["#travel", "#wanderlust", "#adventure", "#explore", "#travelgram",
                "#instatravel", "#trip", "#vacation", "#travelphotography", "#journey"],
    "fashion": ["#fashion", "#style", "#ootd", "#outfitoftheday", "#lookoftheday",
                "#fashionista", "#streetstyle", "#trend", "#clothes", "#instafashion"],
    "indoor" : ["#indoorvibes", "#hometime", "#cozy", "#insidestories", "#roomtour",
                "#interior", "#lifestyle", "#dailylife", "#home", "#comfort"],
    "general": ["#instagood", "#photooftheday", "#picoftheday", "#instadaily", "#life",
                "#photography", "#happy", "#love", "#beautiful", "#instagram"],
}

# ── Emoji banks ───────────────────────────────────────────────────────────────
EMOJIS = {
    "people" : ["✨", "🌟", "💫", "😊", "🙌", "💕", "🔥", "👑", "💯", "🫶"],
    "nature" : ["🌿", "🌊", "🏔️", "🌅", "☀️", "🌸", "🍃", "🌺", "🦋", "🌈"],
    "food"   : ["🍽️", "😋", "🍴", "❤️", "🔥", "🍕", "🥗", "🍜", "🧁", "🍓"],
    "sports" : ["💪", "🏆", "🔥", "⚡", "🎯", "🥊", "🏅", "⚽", "🎽", "🦾"],
    "travel" : ["✈️", "🗺️", "🧭", "🌍", "📍", "🏖️", "🗼", "🌄", "🚀", "🛤️"],
    "fashion": ["👗", "💅", "✨", "🤍", "💎", "👠", "🕶️", "💄", "🎀", "👒"],
    "indoor" : ["🏠", "☕", "📚", "🕯️", "🛋️", "🎵", "🌙", "💡", "🎨", "🧸"],
    "general": ["✨", "💫", "🌟", "❤️", "🙌", "🔥", "💯", "🎉", "💖", "🌈"],
}

# ── Instagram caption templates ───────────────────────────────────────────────
INSTAGRAM_CAPTIONS = {
    "people_woman": [
        "She believed she could so she did {e}",
        "Confidence is my best outfit {e}",
        "Her vibe speaks louder than words {e}",
        "Own your story {e}",
        "She is a whole vibe {e}",
        "Not just a pretty face {e}",
        "Unbothered and thriving {e}",
        "Too blessed to be stressed {e}",
        "She who dares wins {e}",
        "Making it look easy {e}",
        "Living proof that dreams come true {e}",
        "Radiating good energy only {e}",
        "Main character energy {e}",
        "That girl era {e}",
        "Soft life activated {e}",
    ],
    "people_man": [
        "Stay focused stay humble {e}",
        "Built different {e}",
        "Making moves silently {e}",
        "Hustle harder dream bigger {e}",
        "The grind never stops {e}",
        "Level up every single day {e}",
        "Unbothered unmoved {e}",
        "Chosen by the universe {e}",
        "Real ones move in silence {e}",
        "Consistency is the key {e}",
    ],
    "people_smiling": [
        "Happiness is a choice {e}",
        "Smiling through it all {e}",
        "Good vibes only {e}",
        "Living my best life {e}",
        "Joy looks good on me {e}",
        "Smile big laugh often {e}",
        "Happiness is homemade {e}",
        "Good mood all day {e}",
        "Blessed and grateful {e}",
        "Laughing through the chaos {e}",
        "Positive mind positive life {e}",
        "Today is a good day {e}",
    ],
    "people_group": [
        "Squad goals achieved {e}",
        "These are my people {e}",
        "Friends who laugh together stay together {e}",
        "Better together {e}",
        "My tribe my vibe {e}",
        "Good times good people {e}",
        "Creating memories with the best ones {e}",
        "Us against the world {e}",
        "Friendships that feel like home {e}",
        "The ones who get it {e}",
    ],
    "nature_general": [
        "Nature is my therapy {e}",
        "Take only memories leave only footprints {e}",
        "Into the wild {e}",
        "Earth has music for those who listen {e}",
        "Breathe in the wild air {e}",
        "Lost in the right direction {e}",
        "Nature never goes out of style {e}",
        "Healing in progress {e}",
        "Back to my roots {e}",
        "The earth is calling {e}",
    ],
    "nature_beach": [
        "Salt in the air sand in my hair {e}",
        "Ocean air salty hair {e}",
        "Life is better at the beach {e}",
        "Vitamin sea {e}",
        "Sandy toes sun kissed nose {e}",
        "Beach please {e}",
        "Current mood beach mode {e}",
        "Good times and tan lines {e}",
        "Let the sea set you free {e}",
    ],
    "nature_mountain": [
        "On top of the world {e}",
        "Higher the mountain stronger the soul {e}",
        "Mountains are calling {e}",
        "Climb mountains not so the world can see you {e}",
        "Peak happiness {e}",
        "Above the clouds {e}",
        "Sky is not the limit {e}",
    ],
    "sports": [
        "No pain no gain {e}",
        "Train hard win easy {e}",
        "Champions are made in the off season {e}",
        "Sweat now shine later {e}",
        "Push your limits {e}",
        "Stronger every day {e}",
        "Earn it {e}",
        "Beast mode on {e}",
        "Work hard play harder {e}",
        "Built not born {e}",
        "Discipline over motivation {e}",
        "Your only competition is yourself {e}",
    ],
    "food": [
        "Good food good mood {e}",
        "Eat well travel often {e}",
        "First I eat then I do everything else {e}",
        "Food is my love language {e}",
        "Life is short eat dessert first {e}",
        "Fueling the soul {e}",
        "Made with love {e}",
        "Happiness is homemade {e}",
        "Calories do not count on weekends {e}",
        "Treat yourself {e}",
    ],
    "travel": [
        "Wanderlust and city dust {e}",
        "Not all those who wander are lost {e}",
        "Adventure awaits {e}",
        "Collect moments not things {e}",
        "Explore more {e}",
        "Travel is the only thing you buy that makes you richer {e}",
        "Catch flights not feelings {e}",
        "New places new faces {e}",
        "The world is yours to explore {e}",
        "Born to roam {e}",
        "Next stop everywhere {e}",
        "Life is short and the world is wide {e}",
    ],
    "fashion_white": [
        "White is always right {e}",
        "Clean looks clean feels {e}",
        "White shirt energy only {e}",
        "All white everything {e}",
        "Keep it clean keep it classic {e}",
        "Simplicity is the ultimate sophistication {e}",
    ],
    "fashion_general": [
        "Dressed up with nowhere to go {e}",
        "Style is a way to say who you are {e}",
        "Fashion is my passion {e}",
        "Outfit on point {e}",
        "Slaying in style {e}",
        "Look good feel good {e}",
        "Confidence is the best outfit {e}",
        "Dress like you are already famous {e}",
        "Fashion fades style is eternal {e}",
        "Own the look {e}",
    ],
    "indoor": [
        "Home is where the heart is {e}",
        "Cozy vibes only {e}",
        "My happy place {e}",
        "Good things happen inside {e}",
        "Indoor soul outdoor goals {e}",
        "Stay home stay sane {e}",
        "Home sweet home {e}",
        "Comfort zone activated {e}",
    ],
    "general": [
        "Making every moment count {e}",
        "Living the dream {e}",
        "Good things take time {e}",
        "Every day is a new beginning {e}",
        "Just here doing my thing {e}",
        "Vibes speak louder than words {e}",
        "Creating my own sunshine {e}",
        "The best is yet to come {e}",
        "Grateful for everything {e}",
        "Life is beautiful {e}",
        "Write your own story {e}",
        "Today is a gift {e}",
        "Magic is everywhere {e}",
        "Keep going keep glowing {e}",
        "Small steps big dreams {e}",
        "Progress not perfection {e}",
        "Choose joy every single day {e}",
        "Be the energy you want to attract {e}",
        "Doing it for the memories {e}",
        "This is my moment {e}",
    ],
}


def detect_theme(caption):
    """
    Detects theme of caption based on keywords.
    Returns detailed theme string.
    """
    caption_lower = caption.lower()

    # food
    if any(w in caption_lower for w in ["eat", "food", "drink", "cook", "meal", "restaurant"]):
        return "food"

    # sports
    elif any(w in caption_lower for w in ["run", "play", "sport", "game", "gym", "jump", "swim", "kick"]):
        return "sports"

    # beach
    elif any(w in caption_lower for w in ["beach", "ocean", "sea", "wave", "sand", "shore"]):
        return "nature_beach"

    # mountain
    elif any(w in caption_lower for w in ["mountain", "hill", "peak", "cliff", "summit"]):
        return "nature_mountain"

    # general nature
    elif any(w in caption_lower for w in ["field", "forest", "park", "sky", "sunset", "tree", "flower", "grass"]):
        return "nature_general"

    # travel
    elif any(w in caption_lower for w in ["travel", "street", "city", "road", "plane", "landmark", "tourist"]):
        return "travel"

    # fashion white
    elif "white" in caption_lower and any(w in caption_lower for w in ["shirt", "dress", "jacket", "clothes", "wear"]):
        return "fashion_white"

    # fashion general
    elif any(w in caption_lower for w in ["shirt", "dress", "wear", "jacket", "clothes", "fashion", "outfit"]):
        return "fashion_general"

    # indoor
    elif any(w in caption_lower for w in ["room", "indoor", "inside", "home", "house", "office", "desk"]):
        return "indoor"

    # smiling people
    elif any(w in caption_lower for w in ["smil", "laugh", "happy", "joy", "grin"]):
        return "people_smiling"

    # woman
    elif any(w in caption_lower for w in ["woman", "girl", "lady", "female", "she", "her"]):
        return "people_woman"

    # man
    elif any(w in caption_lower for w in ["man", "boy", "guy", "male", "he", "his"]):
        return "people_man"

    # group
    elif any(w in caption_lower for w in ["group", "people", "crowd", "team", "friends", "together", "children"]):
        return "people_group"

    else:
        return "general"


def make_instagram_caption(raw_caption):
    """
    Converts raw model output to Instagram style caption.
    """
    # detect detailed theme
    theme = detect_theme(raw_caption)

    # get emoji bank for broad theme
    broad_theme = theme.split("_")[0] if "_" in theme else theme
    emoji_bank  = EMOJIS.get(broad_theme, EMOJIS["general"])
    emoji       = random.choice(emoji_bank)

    # get hashtag bank
    hashtag_bank = HASHTAGS.get(broad_theme, HASHTAGS["general"])
    hashtags     = random.sample(hashtag_bank, min(4, len(hashtag_bank)))

    # get caption templates for this theme
    templates = INSTAGRAM_CAPTIONS.get(theme, INSTAGRAM_CAPTIONS["general"])

    # pick random template and fill emoji
    template = random.choice(templates)
    caption  = template.format(e=emoji)

    # combine caption + hashtags
    final = f"{caption} {' '.join(hashtags)}"

    return final


def make_linkedin_caption(raw_caption):
    """
    Converts raw caption to LinkedIn professional style.
    """
    templates = [
        "Every experience shapes who we become. Grateful for every moment in this journey. #Growth #Professional #Mindset",
        "Success is not just about what you accomplish but what you inspire others to do. #Leadership #Inspiration",
        "Showing up every day with purpose and passion. That is the real secret. #WorkEthic #Goals #Success",
        "The best investment you can make is in yourself. Keep learning keep growing. #PersonalDevelopment #Growth",
        "Behind every great achievement is a story of dedication and perseverance. #Motivation #Success",
        "Great things never come from comfort zones. Keep pushing keep growing. #Ambition #Leadership",
        "Your network is your net worth. Invest in meaningful connections. #Networking #Professional",
        "Consistency beats talent when talent is not consistent. Show up every day. #Discipline #Success",
        "The future belongs to those who believe in the beauty of their dreams. #Vision #Goals",
        "Work hard in silence let your success make the noise. #Hustle #Achievement",
    ]
    return random.choice(templates)


def make_twitter_caption(raw_caption):
    """
    Converts raw caption to Twitter punchy style.
    """
    theme      = detect_theme(raw_caption)
    broad      = theme.split("_")[0] if "_" in theme else theme
    emoji      = random.choice(EMOJIS.get(broad, EMOJIS["general"]))

    options = [
        f"no thoughts just {emoji}",
        f"main character energy {emoji}",
        f"this is the vibe {emoji}",
        f"not me actually doing things {emoji}",
        f"living rent free in my own life {emoji}",
        f"told myself I would and I did {emoji}",
        f"the audacity to slay every day {emoji}",
        f"it is giving everything {emoji}",
        f"the moment we have been waiting for {emoji}",
        f"POV you are thriving {emoji}",
    ]
    return random.choice(options)


def make_email_caption(raw_caption):
    """
    Converts raw caption to professional email style.
    """
    clean = raw_caption.strip().rstrip('.')
    if len(clean) < 3:
        clean = "the attached image"
    else:
        clean = clean[0].upper() + clean[1:]

    templates = [
        f"{clean}. Please find the attached image for your reference.",
        f"Please see the attached image showing {clean.lower()}.",
        f"Attached is an image of {clean.lower()} for your review.",
    ]
    return random.choice(templates)


def generate_platform_caption(raw_caption, platform):
    """
    Main function — takes raw model output and platform
    returns polished platform specific caption.
    """
    # handle empty or very short raw captions
    if not raw_caption or len(raw_caption.strip()) < 3:
        raw_caption = "a person in a room"

    if platform == "instagram":
        return make_instagram_caption(raw_caption)
    elif platform == "linkedin":
        return make_linkedin_caption(raw_caption)
    elif platform == "twitter":
        return make_twitter_caption(raw_caption)
    elif platform == "email":
        return make_email_caption(raw_caption)
    else:
        return raw_caption