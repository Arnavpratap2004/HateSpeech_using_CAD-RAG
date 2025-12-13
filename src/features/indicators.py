"""
Comprehensive Hate Speech Indicators Dictionary
================================================
This module contains an extensive dictionary of keywords and phrases
used to detect various categories of hate speech.

Usage:
    from indicators import HATE_SPEECH_INDICATORS
"""

HATE_SPEECH_INDICATORS = {
    # Caste-based hate (India-specific)
    "Caste-based hate": [
        "lower castes", "upper castes", "dalit", "brahmin", "caste", "untouchable",
        "chamar", "bhangi", "scheduled caste", "scheduled tribe", "obc", "savarna",
        "shudra", "kshatriya", "vaishya", "outcaste", "caste system", "reservation",
        "quota", "casteist", "manuwadi", "varna", "jati", "gotra", "sub-caste",
        "intercaste", "caste pride", "born into", "by birth", "bloodline", "ancestry"
    ],

    # Religion-based hate
    "Religion-based hate": [
        "muslim", "islam", "hindu", "christian", "jew", "sikh", "buddhist", "religion",
        "allah", "jesus", "mosque", "temple", "church", "synagogue", "gurdwara",
        "quran", "bible", "gita", "torah", "kafir", "infidel", "convert", "jihad",
        "crusade", "fundamentalist", "extremist", "terrorist", "radical islam",
        "islamophobia", "antisemitic", "zionist", "hindutva", "evangelist",
        "blasphemy", "heretic", "pagan", "heathen", "godless", "atheist", "agnostic",
        "mullah", "priest", "rabbi", "imam", "religious fanatic", "burqa", "hijab",
        "skull cap", "turban", "religious symbols", "halal", "kosher", "beef eater",
        "pork eater", "idol worshipper", "cow worshipper", "love jihad", "forced conversion"
    ],

    # Gender-based hate
    "Gender-based hate": [
        "women belong", "men are superior", "feminist", "sexist", "women", "men", "gender",
        "feminazi", "misogynist", "misandrist", "patriarchy", "toxic masculinity",
        "weak sex", "fair sex", "gold digger", "slut", "whore", "bitch", "hoe",
        "simp", "incel", "cuck", "beta male", "alpha male", "real man", "real woman",
        "kitchen", "sandwich", "barefoot", "pregnant", "housewife", "trophy wife",
        "man up", "grow a pair", "like a girl", "throw like a girl", "cry like a girl",
        "emotional", "hysterical", "nagging", "bossy", "frigid", "easy",
        "asking for it", "dress code", "what were you wearing", "friendzone",
        "not like other girls", "pick me", "thot", "harlot", "jezebel"
    ],

    # LGBTQ+ hate
    "LGBTQ+ hate": [
        "gay", "lesbian", "homosexual", "bisexual", "transgender", "trans", "queer",
        "faggot", "dyke", "tranny", "shemale", "ladyboy", "he-she", "it", "pervert",
        "deviant", "unnatural", "against nature", "abomination", "sodomy", "sodomite",
        "confused", "mental illness", "choose to be", "lifestyle choice", "gay agenda",
        "groomer", "groom children", "drag queen", "cross-dresser", "non-binary",
        "pronoun", "deadname", "detransition", "conversion therapy", "pray the gay away",
        "don't say gay", "lgbtq propaganda", "rainbow flag", "pride parade"
    ],

    # Race and ethnicity-based hate
    "Race-based hate": [
        "n-word", "negro", "colored", "black people", "white people", "brown people",
        "yellow", "redskin", "wetback", "beaner", "spic", "chink", "gook", "slant-eye",
        "curry", "pajeet", "streetshitter", "gangster", "thug", "ghetto", "trailer trash",
        "white trash", "redneck", "hillbilly", "illegal alien", "foreigner", "immigrant",
        "go back", "your country", "not from here", "outsider", "ethnic", "mixed race",
        "half breed", "mulatto", "mongrel", "pure blood", "pure race", "master race",
        "racial purity", "white power", "black power", "supremacy", "supremacist",
        "slavery", "colonizer", "colonized", "uncivilized", "savage", "primitive", "tribal"
    ],

    # Nationality and xenophobia
    "Nationality-based hate": [
        "go back to your country", "illegal", "alien", "foreigner", "immigrant",
        "migrant", "refugee", "asylum seeker", "border", "deportation", "invasion",
        "infestation", "swarm", "horde", "stealing jobs", "our country", "our land",
        "nationalist", "patriot", "anti-national", "traitor", "sedition",
        "pakistani", "chinese", "american", "british", "african", "arab",
        "terrorist country", "shithole country", "third world", "backward",
        "developing nation", "first world", "western values", "eastern barbarism",
        "cultural invasion", "replacing us", "great replacement", "open borders"
    ],

    # Political hate
    "Political hate": [
        "political", "government", "traitor", "election", "congress", "regime",
        "leftist", "rightist", "liberal", "conservative", "communist", "fascist",
        "nazi", "socialist", "capitalist", "marxist", "anarchist", "extremist",
        "radical", "woke", "snowflake", "libtard", "conservatard", "bhakt",
        "andh-bhakt", "anti-national", "presstitute", "paid media", "godi media",
        "propaganda", "fake news", "stolen election", "deep state", "establishment",
        "elite", "globalist", "shill", "bootlicker", "puppet", "sold out",
        "vote bank", "appeasement", "minority appeasement", "majority oppression"
    ],

    # Cyberbullying
    "Cyberbullying": [
        "online", "internet", "dox", "troll", "cyber", "twitter", "facebook",
        "instagram", "tiktok", "reddit", "screenshot", "expose", "cancel",
        "cancelled", "ratio", "ratioed", "block", "report", "harassment",
        "stalking", "cyberstalking", "revenge porn", "leaked", "nudes",
        "kill yourself", "kys", "die", "death threats", "rape threats",
        "swatting", "doxxing", "ip address", "home address", "personal info",
        "brigade", "dogpile", "pile on", "mass report", "ban evasion",
        "fake account", "burner account", "anonymous", "coward", "keyboard warrior"
    ],

    # Subtle hate and dog whistles
    "Subtle hate": [
        "you know the type", "those kind", "just a joke", "can't take a joke",
        "i'm not racist but", "i have black friends", "some of my best friends",
        "no offense but", "just saying", "truth hurts", "facts don't care",
        "playing the victim", "race card", "gender card", "oppression olympics",
        "virtue signaling", "politically correct", "thought police", "cancel culture",
        "free speech", "just an opinion", "different opinion", "agree to disagree",
        "both sides", "devil's advocate", "hypothetically", "just asking questions",
        "do your research", "wake up", "sheeple", "red pilled", "based",
        "triggered", "safe space", "echo chamber", "bubble", "mainstream media"
    ],

    # Disability-based hate
    "Disability-based hate": [
        "retard", "retarded", "mentally ill", "crazy", "insane", "psycho",
        "schizo", "bipolar", "autistic", "spastic", "cripple", "invalid",
        "handicapped", "disabled", "differently abled", "special needs",
        "short bus", "window licker", "dumb", "deaf", "blind", "mute",
        "wheelchair", "asylum", "mental hospital", "institution", "normal people",
        "birth defect", "genetic defect", "burden on society", "useless eater"
    ],

    # Age-based hate
    "Age-based hate": [
        "boomer", "ok boomer", "millennial", "gen z", "zoomer", "old people",
        "young people", "elderly", "senior citizen", "geriatric", "senile",
        "dementia", "out of touch", "dinosaur", "fossil", "ancient",
        "kids these days", "back in my day", "respect your elders",
        "entitled generation", "lazy generation", "snowflake generation",
        "participation trophy", "tide pod", "avocado toast"
    ],

    # Body shaming
    "Body-based hate": [
        "fat", "obese", "overweight", "skinny", "anorexic", "bulimic",
        "ugly", "disgusting", "gross", "pig", "whale", "cow", "skeleton",
        "short", "midget", "dwarf", "giant", "freak", "bald", "balding",
        "body shape", "body type", "real women", "real men", "beach body",
        "diet", "gym", "workout", "lazy", "no self-control", "gluttony"
    ],

    # Economic and class-based hate
    "Class-based hate": [
        "poor people", "rich people", "homeless", "beggar", "lower class",
        "upper class", "middle class", "working class", "elite", "bourgeoisie",
        "proletariat", "poverty", "welfare queen", "food stamps", "moocher",
        "leech", "parasite", "freeloader", "taxpayer money", "handout",
        "bootstraps", "self-made", "privileged", "underprivileged", "slum",
        "ghetto", "hood", "projects", "trust fund", "silver spoon", "entitled"
    ],

    # Dehumanization terms
    "Dehumanization": [
        "animal", "beast", "monster", "vermin", "pest", "cockroach", "rat",
        "snake", "dog", "pig", "ape", "monkey", "subhuman", "less than human",
        "not people", "not human", "creature", "thing", "it", "filth",
        "garbage", "trash", "waste", "scum", "cancer", "disease", "plague",
        "infestation", "exterminate", "eradicate", "cleanse", "purge", "genocide"
    ],

    # Violence and threat indicators
    "Violence indicators": [
        "kill", "murder", "death", "die", "hang", "shoot", "stab", "beat",
        "attack", "assault", "rape", "lynch", "burn", "bomb", "explode",
        "destroy", "eliminate", "execute", "slaughter", "massacre", "genocide",
        "ethnic cleansing", "final solution", "holocaust", "pogrom", "purge",
        "deserve to die", "should be killed", "hope you die", "wish death upon",
        "bullet", "gun", "knife", "weapon", "violence", "violent", "force"
    ],

    # Historical hate references
    "Historical hate references": [
        "nazi", "hitler", "holocaust", "concentration camp", "gas chamber",
        "auschwitz", "swastika", "aryan", "third reich", "final solution",
        "kkk", "ku klux klan", "confederate", "slavery", "jim crow",
        "apartheid", "segregation", "ethnic cleansing", "genocide", "pogrom",
        "crusade", "inquisition", "witch hunt", "partition", "riots"
    ]
}


def get_all_indicators() -> list:
    """Returns a flat list of all indicators."""
    all_terms = []
    for category_terms in HATE_SPEECH_INDICATORS.values():
        all_terms.extend(category_terms)
    return all_terms


def get_categories() -> list:
    """Returns a list of all hate speech categories."""
    return list(HATE_SPEECH_INDICATORS.keys())


def check_text_for_indicators(text: str) -> dict:
    """
    Check text against all indicators and return matches.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with categories as keys and matched indicators as values
    """
    text_lower = text.lower()
    matches = {}
    
    for category, indicators in HATE_SPEECH_INDICATORS.items():
        found = [ind for ind in indicators if ind in text_lower]
        if found:
            matches[category] = found
            
    return matches


def get_indicator_count() -> dict:
    """Returns the count of indicators per category."""
    return {cat: len(terms) for cat, terms in HATE_SPEECH_INDICATORS.items()}


# Summary statistics
if __name__ == "__main__":
    print("=" * 50)
    print("HATE SPEECH INDICATORS SUMMARY")
    print("=" * 50)
    
    counts = get_indicator_count()
    total = sum(counts.values())
    
    for category, count in counts.items():
        print(f"  {category}: {count} indicators")
    
    print("-" * 50)
    print(f"  TOTAL: {total} indicators across {len(counts)} categories")
    print("=" * 50)
