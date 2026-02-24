import re
import sys
from lingua import Language, LanguageDetectorBuilder

# encodage
sys.stdout.reconfigure(encoding="utf-8")

# Initialisation du détecteur Lingua 
languages = [Language.ENGLISH, Language.FRENCH, Language.ARABIC]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# NETTOYAGE DU TEXTE
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    # Garder lettres latines + lettres arabes + chiffres + espaces + accents français
    text = re.sub(r"[^a-z0-9\u0600-\u06FF\s\u00C0-\u017F]", " ", text)
    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text

# DÉTECTION DES LANGUES
def detect_language(text):
    clean_t = clean_text(text)
    if not clean_t:
        return "unknown"

    # Vérification des scripts
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', clean_t))
    has_latin = bool(re.search(r'[a-z]', clean_t))

    # CAS 1 : MIXED (Arabe + Latin) 
    if has_arabic and has_latin:
        return "mixed"

    # CAS 2 : TEXTE LATIN 
    if has_latin and not has_arabic:
        # Dictionnaires de mots-clés 
        fr_keywords = [
    "je", "suis", "me", "sens", "très", "mal", "est", "c'est", "dois","tu", "il", "elle", "nous", "vous", "ils", "elles", "on", "moi", "toi",
    "ai", "as", "a", "à", "sommes", "êtes", "sont", "vais", "vas", "va", "font", "peux", "veut", "crois", "pense", "ressens", "souffre", "travaille",
    "avec", "sans", "pour", "dans", "chez", "mais", "donc", "parce que", "car", "aussi", "encore", "déjà", "toujours", "jamais",
    "fatigué", "épuisé", "nerveux", "inquiet", "triste", "heureux", "calme", "occupé", "énervé", "tendu",
    "il y a", "je dois", "j'ai besoin", "tout le temps", "parfois", "peut-être", "bien sûr",
    "un peu", "vraiment", "aujourd’hui", "maintenant", "stressé", "stressée", "pression", "sous pression",
    "dépassé", "anxieux", "débordé", "détendu", "léger", "travail", "service", "responsabilités", "tâches",
    "deadlines", "manque de temps", "temps pour moi", "organiser", "pause", "avenir", "futur",
    "situation", "quotidien", "arrive", "essaie", "ça va", "manageable", "sous contrôle", "pour l’instant",
    "occupé", "maison", "bureau", "sous pression", "ne peut pas se détendre", "ne peut pas se calmer", "gérable", "gérable", "bien",
    "paisible", "ça va", "clair", "esprit", "essayer", "encore", "aujourd’hui", "surchargé", "charge de travail",
    "projets", "pause", "pause", "carrière", "avenir", "niveau de stress", "relaxation", "moment",
    "légèrement", "un peu", "gérable", "gestion", "gérer", "colère", "avec colère", "colère", "en colère",
    "agacer", "agacement", "contrariétés", "agacé", "agaçant", "antisocial", "anxiétés", "anxiété", "anxieux", "anxieusement", "anxiété",
    "blâmer", "blocage", "imbécile", "pause", "rupture", "ruptures", "effondrement", "cassant", "pauses", "rupture", "ruptures",
    "étouffer", "se plaindre", "s’est plaint", "se plaignant", "se plaint", "plainte", "plaintes", "complexe", "compliqué", "complication",
    "confronter", "confrontation", "conflictuel", "confondre", "confus", "confond", "confusant",
    "tordu", "pleurer", "bon sang", "dommage", "endommagé", "dommages", "nuisible", "danger", "dangereux", "dangerosité", "sombre",
    "assombrir", "assombri", "plus sombre", "obscurité", "mort", "déclin", "déclins", "en déclin", "dur", "difficile", "difficultés", "difficulté",
    "manque de respect", "méprisable", "caractère méprisable", "irrespectueux", "irrespectueusement", "irrespect", "manquant de respect",
    "détresse", "en détresse", "angoissant", "urgence", "excuse", "excuses", "épuiser", "épuisé",
    "échouer", "échoué", "en train d’échouer", "échoue", "échec", "échecs", "faute", "fautes", "peur", "craintif", "avec peur", "peurs",
    "paniquer", "paniquant", "haine", "détesté", "haineux", "haineusement", "haine", "personne haineuse", "personnes haineuses", "déteste", "détestant",
    "mal de tête", "maux de tête", "briseur de cœur", "déchirant", "de façon déchirante", "sans cœur",
    "blessé", "blessant", "faisant mal", "fait mal", "idiot", "invalide", "manque", "folie", "fou", "nier", "négation", "négatif", "points négatifs", "négativité", "négliger", "négligé",
    "nerveux", "nerveusement", "douleur", "douloureux", "douloureux", "problème", "problématique", "problèmes", "refuser", "refusé", "refuse", "refusant",
    "regret", "triste", "tristement", "tristesse", "fatigué", "travail", "au top", "remercier", "reconnaissant", "super", "succès", "succès", "succès", "réussi", "avec succès"
]

        en_keywords = [
    "i", "feel", "am", "very", "today", "depressed", "stressed", "tired", "too", "much", "have",
    "happy", "me", "my", "mine", "you", "your", "he", "she", "it", "we", "they",
    "do", "did", "does", "will", "can", "could", "should",
    "anxious", "worried", "exhausted", "burned", "sad", "angry", "relaxed", "overwhelmed",
    "pressure", "nervous", "lonely", "upset","depressed","depressing","depressingly","depression","depressions",
    "and", "but", "because", "since", "now", "yesterday", "tomorrow",
    "always", "never", "often", "sometimes", "with", "without", "about",
    "work", "job", "task", "tasks", "life", "people", "someone", "everything", "nothing", "something",
    "busy", "home", "office","under pressure", "can’t relax", "can’t calm down", "manageable", "manageable", "fine",
    "peaceful", "okay", "clear", "mind", "trying", "still", "today", "overloaded", "workload",
    "projects", "break", "pause", "career", "future", "stress level", "relaxation", "moment",
    "slightly", "bit", "manageable", "handling", "handle", "anger", "angrily", "angriness", "angry",
    "annoy", "annoyance", "annoyances", "annoyed", "annoying","anti-social","anxieties", "anxiety", "anxious", "anxiously", "anxiousness",
    "blame","blockage", "blockhead","break","break-up","break-ups","breakdown","breaking","breaks","breakup","breakups",
    "choke","complain","complained","complaining","complains","complaint","complaints","complex","complicated","complication",
    "confront","confrontation","confrontational","confuse","confused","confuses","confusing",
    "crooked","cry","d*mn","damage","damaged","damages","damaging","danger","dangerous","dangerousness","dark",
    "darken","darkened","darker","darkness","dead","decline","declines","declining","hard","difficult","difficulties","difficulty",
    "disrespect","disrespectable","disrespectablity","disrespectful","disrespectfully","disrespectfulness","disrespecting",
    "distress","distressed","distressing","emergency","excuse","excuses","exhaust","exhausted",
    "fail","failed","failing","fails","failure","failures","fault","faults","fear","fearful","fearfully","fears",
    "freak","freaking","hate","hated","hateful","hatefully","hatefulness","hater","haters","hates","hating",
    "headache","headaches","heartbreaker","heartbreaking","heartbreakingly","heartless",
    "hurted","hurtful","hurting","hurts","idiot","invalid","lack","madness","mad","negate","negation","negative","negatives","negativity","neglect","neglected",
    "nervous","nervously","pain","painful","painfull","problem","problematic","problems","refuse","refused","refuses","refusing",
    "regret","sad","sadly","sadness","tired","work","top","thank","thankful","super","succes","success","successes","successful","successfully",



]

        darija_lat_keywords = [
    "bzzaf", "kanhess", "shwiya", "khadma", "dyali", "3andi", "merta7", "brasi", "eliya", "rassi",
    "m9alla9", "mqalaq", "m3asseb", "m3asab", "3iyan", "3iyana", "mkhno9", "mkhnoq", "mfoussi", "mroubel",
    "t3telt", "moushkil", "mouchkil", "khassni", "bghit", "ndir", "nrtah", "tneffess", "kandir", "ghadi", "nmchi", "nqdar", "nsali", "nkhdm",
    "daba", "lyouma", "dima", "mrra", "ba9i", "mazal", "ga3", "koulchi", "walou", "chwiya", "bezeff", "bezef",
    "ana", "nta", "nti", "dialek", "dialo", "dyalk", "3andek", "3ando", "wach", "fin", "3lach", "kifach",
    "bikhir","labass","mashi", "mhelook", "mqala9", "mbrzt", "mstresiya", "kalm", "khfif", "frhana",
    "mfr7a", "bikhir", "ta 7aja", "kolchi mzyan", "l3amal", "ga3","3aDim",
    "3aamr","aamr","3amer","fr7an","far7an","7azin",
    "mchghoul", "mchghol", "mechghoul", "machghoul", "mchton", "mchtoun", "mechTon", "machToun","m3ssb", "m3eseb",
    "mrwn", "mrewen","mrawen","3yyan" , "3eyyan", "mtfa2l", "mtfa2el", "motafa2il","mtcha2m","mtcha2em","9ala9","t9li9a","qalaq",
    "3tbrziT","tberziT","tbarziT","ghaDab","nachaT", "nachaaT","ka2aaba", "ka2aba","khouf", "5ouf", "khaouf", "5aouf",
    "i7baT","7amas",  "hamas","sa3ada", "saaada", "lfer7a","fara7","farah","chaghaf","mo3anat", "mo3aanaat",
    "khl3a", "khel3a","khal3a", "5al3a","cho3our", "cho3or",
    "tania", "thania", "d9i9a", "daqiqa", "da9i9a", "yawm", "youm", "nhar", "lbar7", "lbare7", "lbareh", "lbarh", "daba", "drok", "lyom", "liom", "lioma", "lyoma",
    "ghdda" ,"ghedda","ghadda","wellbar7","wallbare7","wllbare7", "b3d ghdda","ba3d ghdda","be3d ghdda", "simana","semana", "chher","chhar","tnin ",
    "tlat","larb3","larba3" ,"larbe3","lkhmis" ,"l5miss" ,"lakhmis" ,"lekhmis", "jm3a" ,"jam3a" ,"jem3a" ,"jjm3a" ,"sbt" ,"ssbt" ,"sabt" ,"ssebt", "l7d" ,"l7ed","l7add" ,"l7edd",
    "wikand" ,"lwikand", "yanayir" ,"yanayr", "ibrayr" ,"fibrayer", "abril", "yonyo", "yolyoz" ,"youlyouz" ,"ghocht", "chotanbir", "oktoubr" ,"oktobr", "nowanbir", "dojanbir" ,"doujanbir",
    "chher wahd","chher wahd",	"chher wa7d",	"chher wa7d", "chher jouj",	"chher jouj",	"chher jouj","chher joj",
    "chher tlata", "chher tlata", "chher tlata", "chher rb3a", "chher rb3a", "chher rb3a", "chher reb3a",
    "chher khmsa", "chher khmsa", "chher khmsa", "chher khamsa", "chher stta", "chher stta", "chher stta",
    "chher sb3a", "chher sb3a", "chher sb3a", "chher seb3a", "chher tmnia", "chher tmnia", "chher tmnia", "chher tmenia",
    "chher ts3oud", "chher ts3oud", "chher ts3oud", "chher ts3eud", "chher 3chra", "chher 3chra", "chher 3chra", "chher 3echra",
    "chher 7Dach", "chher 7Dach", "chher 7Dach", "chher Tnach", "chher Tnach", "chher Tnach","lil","ftra",	"fatra", "fetra","teasebt","t3sbt","tesebt",
    "nas", "chi wahed", "koulchi", "walou "," hta haja", "chi haja", "mechghoul", "dar", "bureau "," mektab",
    "medghout", "mane9derch nrtah", "ma9dertch ntheden", "mezyan "," bikhir",
    "hani "," merta7", "cava "," mezyan", "safi "," wad7", "3qel ", "kan7awel", "mazal "," baqi",
    "lyouma", "3amer ", "khedma ktira ", "machari3 ", "istiraha ","wa9fa", "mosta9bal", "ra7a",
    "dehka "," la7da", "chwiya", "shwiya ", "kant3amel m3a","net3amel ", "lghadab "," l3asabiya", "blghadab "," bl3asabiya",
    "lghadab", "m3asseb "," ghadban", "berzetni "," tsedde3", "iz3aj "," tseda3t","sda3"
    "iz3ajat "," mchakil", "mberzet "," mqelle9", "kiyberzet "," mostafiz",
    "ma kiykhltch "," mashi jtima3i", "qala9 "," mkhelet","mekhlou3 "," mstressé", "blkhouf "," blqala9", "lkhouf ",
    "louma", "tblockit ", "meklakh "," ras","hras ", "tfarqo", "tfari9", "inhiyar ",
    "tehras", "therissat", "tfari9", "tfari9","tkhneqt "," chka", "tchiyka",
    "kitchka", "kitchiykaw", "chkaya", "chkayat","m3e9ed", "m3e9ed", "ta3qid"
]

        # Vérification 
        count_fr = 1 if any(re.search(rf"\b{w}\b", clean_t) for w in fr_keywords) else 0
        count_en = 1 if any(re.search(rf"\b{w}\b", clean_t) for w in en_keywords) else 0
        count_darija = 1 if any(re.search(rf"\b{w}\b", clean_t) for w in darija_lat_keywords) else 0

        # MIXED : Si mélange de deux langues latines ou plus
        if (count_fr + count_en + count_darija) >= 2:
            return "mixed"
        
        # Règle DARIJA LATINE 
        if count_darija:
            return "darija_lat"

        # LINGUA pour Français et Anglais
        confidence_values = detector.compute_language_confidence_values(clean_t)
        
        # Si Lingua est incertain on retourne mixed
        if len(confidence_values) > 1 and confidence_values[1].value > 0.25:
            return "mixed"
            
        prediction = detector.detect_language_of(clean_t)
        if prediction == Language.FRENCH:
            return "french"
        elif prediction == Language.ENGLISH:
            return "english"
        else:
            return "english" # Fallback par défaut pour le latin

    # CAS 3 : TEXTE ARABE 
    if has_arabic and not has_latin:
        darija_ar_keywords = [
    "كنحس", "بزاف", "شوية", "ديالي", "راسي", "مقلق", "مرتاح","مزيار", "مخنوق", "معصب", "عيان",
    "عيانة", "مرون", "تالف", "حماق", "فقايص", "مهموم","خاصني", "بغييت", "ندير", "نمشي", "نسالي", 
    "نخدم", "نرتتاح", "غادي", "كاندير", "كنشوف","دابا", "ليوما", "ديما", "مازال", "باقي", "كولشي", "والو", 
    "غدا", "البارح", "مرة","ديالك", "ديالو", "ديالنا", "واش", "علاش", "كيفاش", "ولكن", "حيت", "بيا", "عليا",
    "مضغوط", "مهلوك", "مبرزط", "مستريسية", "كالم", "الارتياح", "خفيف", "فرحان","مفرح", "بخير", "تا حاجة",
    "كلشي مزيان", "الخدمة", "العمل","المشاريع", "المسؤوليات", "تنظيم الوقت",
    "تانية","دقيقة","ساعة","يوم","دابا","ليوم","غدا","بعد غدا","سيمانة","شهر",
    "عام","تنين","تلات","لاربع","لخميس","جمعة","سبت","لحد","ويكاند","يناير","فبراير","أبريل","ماي","شهر واحد",
    "شهر جوج","شهر تلاتة","شهر ربعة","شهر خمسة","شهر ستة","شهر سبعة","شهر تمنية","شهر تسعود","شهر عشرة","شهر حضاش",
    "شهر طناش","وقت","عشية","وسط نهار","فترة","ماڭانة","مناسبة","تولوت ساعة","تبرزيط","نشاط","فقسة","صداع",
    "الضيقا","حريق راس","لافاك","زيرو","فلوس","كريدي","بانكا","لمعيشة","ماحال","عجز","ماشي جتماعي","خداما","خدام",
    "معقد","إزعاج","صداع","مبرزط","شويا","بزاف","كتير","قليل","لخوف","لخلعا","راسي كايطنطن","غير مع راسي","ماحامل",
    "ماحاملة","ندير","حتا حاجا","مبرزطة","عصباتني","ماعنديش الوقت ","ماعنديش الوقت فين نحك راسي","موصيبا","تبلوكيت","صاڢا",
    "صافا","مستريسي","داكشي","هادشي","ديالو","ديالك","ديالي","ديالنا","عقلت","عليها","عليه","عليهم","شي حاجا","حتا حاجا",
    "برزطني","منقدرش","نرتاح","مفيهش","كولشي","تعرم","مقدرتش نتهدن","مقدرتش نرتاح","كنحاول","جهدي",
    "بخير","هانيا","مزيان","صافي","واضح","مشغول","دار","نآشاط","نآشط","متضايق","متأزم","مقهور",
    "مشوش","محير","مرتاح شوية","متراكم","مطفي","مهلوك بزاف","طالع لي ف راسي","معيا","مسدود","الخدمة واقفة","الخدمة زايدة",
    "مشروع واقف","ما مسوقليش","داير اللي عليا","دابا دابا","شوية بشوية","دغيا","فهاذ الوقت",
    "واحد النهار","نهار كامل","كنخمّم","كنحسب","كنقلّب","كنقرّر","كنراجع","كنصبر","كنتهرب","متفاهم","ما متفاهمش","متخاصم",
    "ما باغيش نهضر","باغي نرتاح","باغي ننعس","فرحان","مرتاح","متصالح مع راسي",
    "كالم","مهني","كلشي غادي مزيان","كنآمن براسي","باغي نطوّر راسي","مقهور","كنبكي","هاني","طالع ليّ المورال","الأمور زوينة",
    "فرحان من قلبي","معذّب","حزين شوية","مرتاح بزاف","الأمور محلولة","فرحان بالنجاح","ديدلاين","الوقت كيجري","مزروب",
    "خاصني نكمّل دابا","خاصني نسالي قبل الوقت","ما بقى ليا وقت","قرب الديدلاين","خاصها تسالي دابا","كنخدم تحت الضغط",
    "خاصني","درت","خاصنا","درنا","عندي","بغييت","باغية","درت اللي عليا","عندي موشكل","ما عنديش حل",

]
        if any(w in clean_t for w in darija_ar_keywords):
            return "darija_ar"
        return "arabic"

    return "unknown"


