"""A synthetic newsroom, in the same raw shape as the production MongoDB export.

Used for three things: the test suite, ``--demo`` mode (so the app runs
without the private dataset) and evaluation fixtures. Every headline, outlet
and person here is invented. The generator deliberately reproduces the mess
of the real export (mixed id shapes, odd date formats, soft-deleted rows,
duplicates) so the ingest path is exercised end to end.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .ingest import RawData

# name -> (english subjects, english actions, english objects, hindi headlines, tags)
_BANK: dict[str, tuple[list[str], list[str], list[str], list[str], list[str]]] = {
    "Politics": (
        [
            "The finance ministry",
            "Opposition leaders",
            "The state cabinet",
            "A parliamentary panel",
            "The election commission",
            "Coalition partners",
            "The chief minister",
            "Lawmakers",
        ],
        [
            "clears",
            "debates",
            "defers a vote on",
            "seeks changes to",
            "defends",
            "walks out over",
            "opens talks on",
            "tables",
        ],
        [
            "the urban housing bill",
            "a new voter roll audit",
            "the farm subsidy overhaul",
            "seat sharing for the state polls",
            "the public transport budget",
            "a campaign finance cap",
            "the water sharing accord",
            "the municipal reform ordinance",
        ],
        [
            "संसद में नए विधेयक पर तीखी बहस",
            "चुनाव आयोग ने मतदाता सूची की जांच शुरू की",
            "मुख्यमंत्री ने बजट प्रस्तावों का बचाव किया",
            "विपक्ष ने सदन से वॉकआउट किया",
            "गठबंधन दलों में सीट बंटवारे पर बातचीत",
        ],
        ["government", "elections"],
    ),
    "Business": (
        [
            "The steel maker",
            "A regional airline",
            "The retail chain",
            "Bond investors",
            "The central bank",
            "A logistics startup",
            "Small exporters",
            "The auto supplier",
        ],
        [
            "reports",
            "forecasts",
            "shrugs off",
            "raises funds after",
            "warns of",
            "bets on",
            "cuts costs amid",
            "expands despite",
        ],
        [
            "a jump in quarterly profit",
            "slower festive demand",
            "rising freight rates",
            "a weaker currency",
            "record credit growth",
            "a surge in rural sales",
            "tighter lending rules",
            "a merger with its rival",
        ],
        [
            "कंपनी का तिमाही मुनाफा बढ़ा, शेयर में तेजी",
            "रिजर्व बैंक ने ब्याज दरें स्थिर रखीं",
            "निर्यातकों ने बढ़ते माल भाड़े पर चिंता जताई",
            "खुदरा बिक्री में त्योहारी मांग से उछाल",
            "स्टार्टअप ने नए निवेश दौर में पूंजी जुटाई",
        ],
        ["corporate news", "financial markets"],
    ),
    "Technology": (
        [
            "A chip designer",
            "The payments app",
            "University researchers",
            "A cloud provider",
            "The handset maker",
            "An open source team",
            "A robotics lab",
            "The telecom operator",
        ],
        ["unveils", "patches", "open sources", "pilots", "delays", "benchmarks", "ships", "tests"],
        [
            "a low power processor",
            "an offline translation model",
            "a satellite broadband link",
            "a privacy dashboard",
            "its battery swap network",
            "a warehouse robot fleet",
            "an identity wallet",
            "a fault tolerant database",
        ],
        [
            "नई चिप से फोन की बैटरी लंबी चलेगी",
            "भुगतान ऐप में सुरक्षा खामी ठीक की गई",
            "शोधकर्ताओं ने ऑफलाइन अनुवाद मॉडल बनाया",
            "दूरसंचार कंपनी ने उपग्रह इंटरनेट का परीक्षण किया",
            "रोबोटिक्स लैब ने गोदाम के लिए रोबोट दिखाया",
        ],
        ["artificial intelligence", "technology"],
    ),
    "Sports": (
        [
            "The national eleven",
            "A teenage sprinter",
            "The league champions",
            "The hockey squad",
            "A veteran spinner",
            "The relay team",
            "The home side",
            "The doubles pair",
        ],
        [
            "clinches",
            "falls short in",
            "rallies to win",
            "withdraws from",
            "sets a record in",
            "books a place in",
            "struggles in",
            "dominates",
        ],
        [
            "the tri series final",
            "the continental qualifiers",
            "a rain hit opener",
            "the indoor championship",
            "the season decider",
            "the under nineteen trials",
            "the night fixture",
            "the knockout round",
        ],
        [
            "टीम ने रोमांचक मुकाबले में सीरीज जीती",
            "युवा धावक ने राष्ट्रीय रिकॉर्ड तोड़ा",
            "हॉकी टीम सेमीफाइनल में पहुंची",
            "स्पिनर ने पांच विकेट लेकर मैच पलटा",
            "बारिश से प्रभावित मैच में मेजबान टीम हारी",
        ],
        ["cricket", "sports"],
    ),
    "Entertainment": (
        [
            "The period drama",
            "A debut director",
            "The streaming series",
            "A playback singer",
            "The film festival",
            "The stage revival",
            "A veteran actor",
            "The animation studio",
        ],
        [
            "tops",
            "divides critics with",
            "announces",
            "wraps filming on",
            "wins praise for",
            "postpones",
            "returns with",
            "teases",
        ],
        [
            "its opening weekend",
            "a three hour finale",
            "an all acoustic album",
            "a sequel",
            "the restored classic",
            "a festival premiere",
            "a surprise cameo",
            "its second season",
        ],
        [
            "नई फिल्म ने पहले सप्ताहांत में रिकॉर्ड कमाई की",
            "गायक ने नया एल्बम जारी किया",
            "वेब सीरीज के दूसरे सीजन की घोषणा",
            "फिल्म समारोह में नवोदित निर्देशक को सम्मान",
            "दिग्गज अभिनेता की मंच पर वापसी",
        ],
        ["movies", "celebrity"],
    ),
    "Health & Wellness": (
        [
            "District hospitals",
            "A vaccine trial",
            "Nutrition experts",
            "The health ministry",
            "A sleep study",
            "Rural clinics",
            "Cardiologists",
            "A screening drive",
        ],
        ["report", "links", "recommend", "expand", "question", "track", "flag", "roll out"],
        [
            "fewer dengue admissions",
            "screen time to poor sleep",
            "a salt reduction plan",
            "telemedicine for hill villages",
            "the cost of new insulin",
            "air quality alerts",
            "early diabetes checks",
            "a mental health helpline",
        ],
        [
            "जिला अस्पतालों में डेंगू के मामले घटे",
            "विशेषज्ञों ने नमक कम खाने की सलाह दी",
            "गांवों में टेलीमेडिसिन सेवा का विस्तार",
            "नींद पर अध्ययन में स्क्रीन टाइम का असर दिखा",
            "स्वास्थ्य मंत्रालय ने नई हेल्पलाइन शुरू की",
        ],
        ["pharma and healthcare", "mental health"],
    ),
    "World News": (
        [
            "Negotiators",
            "The island nation",
            "Border officials",
            "A UN envoy",
            "Aid agencies",
            "The trade bloc",
            "Coastal cities",
            "Foreign ministers",
        ],
        ["agree on", "reject", "resume talks over", "brace for", "appeal for", "sign", "suspend", "extend"],
        [
            "a ceasefire timetable",
            "the shipping lane dispute",
            "a prisoner exchange",
            "the monsoon flood response",
            "a grain export corridor",
            "new visa rules",
            "the climate finance pledge",
            "a joint naval patrol",
        ],
        [
            "वार्ताकार संघर्ष विराम की समयसीमा पर सहमत",
            "सहायता एजेंसियों ने बाढ़ राहत की अपील की",
            "विदेश मंत्रियों ने व्यापार समझौते पर हस्ताक्षर किए",
            "सीमा विवाद पर बातचीत फिर शुरू",
            "संयुक्त राष्ट्र दूत ने शांति प्रस्ताव रखा",
        ],
        ["conflicts & war", "international trade"],
    ),
    "Science": (
        [
            "Astronomers",
            "A lunar orbiter",
            "Marine biologists",
            "The space agency",
            "Glacier researchers",
            "A physics team",
            "Field ecologists",
            "The observatory",
        ],
        ["map", "detects", "catalogue", "schedules", "measure", "confirms", "rediscover", "records"],
        [
            "a slow spinning pulsar",
            "ice beneath a polar crater",
            "a deep reef nursery",
            "a reusable booster test",
            "the fastest melt in a decade",
            "a rare particle decay",
            "a frog thought extinct",
            "an unusually bright comet",
        ],
        [
            "अंतरिक्ष एजेंसी ने बूस्टर परीक्षण की तारीख तय की",
            "वैज्ञानिकों ने ध्रुवीय गड्ढे में बर्फ खोजी",
            "खगोलविदों ने नया धूमकेतु देखा",
            "ग्लेशियर पिघलने की रफ्तार दशक में सबसे तेज",
            "समुद्री जीवविज्ञानियों ने नई प्रवाल भित्ति दर्ज की",
        ],
        ["space", "research"],
    ),
}

_OUTLETS = [
    ("Harbour Ledger", 420),
    ("The Meridian Post", 1900),
    ("Civic Wire", 5200),
    ("Northline Dispatch", 9800),
    ("Daily Quorum", 21000),
    ("Lantern Times", 47000),
    ("Samachar Dhara", 6400),
    ("Jan Vaani", 15500),
    ("Praatah Patrika", 88000),
    ("Signal & Field", 260000),
    ("The Tidewater", 730000),
    ("Gram Sandesh", 2400000),
]
_HINDI_PLACES = ["दिल्ली", "मुंबई", "लखनऊ", "पटना", "जयपुर", "भोपाल", "कानपुर", "इंदौर", "नागपुर", "रांची"]
# Invented towns and people, so each synthetic story has words of its own.
_PLACES = [
    "Varunpur",
    "Kestrel Bay",
    "Amberley",
    "Sonagiri",
    "Halden Cross",
    "Pinewick",
    "Teshwar",
    "Marrow Point",
    "Juniper Falls",
    "Ravelstok",
    "Indravati",
    "Copperhill",
    "Saltmere",
    "Ottergate",
    "Bhadrakul",
    "Wrenford",
    "Lumbridge Vale",
    "Nayanagar",
    "Greyhaven",
    "Tamsin Quay",
    "Vellora",
    "Ashcombe",
    "Kirtipur",
    "Dunmore Reach",
]
_FIRST = [
    "Asha",
    "Rohan",
    "Meera",
    "Kabir",
    "Ines",
    "Tomas",
    "Leela",
    "Arjun",
    "Noor",
    "Dev",
    "Priya",
    "Owen",
    "Zara",
    "Ishaan",
]
_LAST = [
    "Varkey",
    "Lindqvist",
    "Desai",
    "Okafor",
    "Mehra",
    "Castell",
    "Iyer",
    "Brandt",
    "Sethi",
    "Halloran",
    "Rao",
    "Quint",
]
_LEDES = [
    "{person} said the change would reach {place} by spring",
    "Around {n} residents of {place} wrote to the council about it, according to {person}",
    "{person} told reporters the {place} pilot had cost less than planned",
    "Records reviewed by {person} show {n} similar cases near {place}",
    "In {place}, {person} described the mood as cautious",
    "{person} expects a second phase after the {place} results",
    "The {place} office confirmed {n} applications, said {person}",
    "Critics in {place}, led by {person}, asked for an independent audit",
]
_HINDI_NAMES = [
    "सुनीता वर्मा",
    "राकेश मिश्रा",
    "अंजलि सिंह",
    "विकास त्रिपाठी",
    "नेहा गुप्ता",
    "अमित यादव",
    "पूजा शर्मा",
    "मनोज तिवारी",
]
_HINDI_OUTLETS = {"Samachar Dhara", "Jan Vaani", "Praatah Patrika", "Gram Sandesh"}
_FILLER = [
    "Officials said the details would be published later this week.",
    "Analysts said the decision had been expected for several months.",
    "Residents interviewed nearby gave a mixed response to the announcement.",
    "A spokesperson declined to comment on the timeline.",
    "The figures were confirmed by two people familiar with the matter.",
    "Further updates are expected after a review meeting on Monday.",
]
_HINDI_FILLER = [
    "अधिकारियों ने कहा कि विस्तृत जानकारी इसी सप्ताह जारी होगी।",
    "विशेषज्ञों के अनुसार यह फैसला कई महीनों से अपेक्षित था।",
    "स्थानीय लोगों ने इस घोषणा पर मिली जुली प्रतिक्रिया दी।",
    "प्रवक्ता ने समयसीमा पर टिप्पणी करने से इनकार किया।",
]


def _oid(*parts: Any) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:24]


def make_raw(n_articles: int = 400, n_users: int = 40, seed: int = 7, now: datetime | None = None) -> RawData:
    """Generate a deterministic raw dataset with ``n_articles`` clean articles."""
    if not 1 <= n_articles <= 3000:
        raise ValueError("n_articles must be between 1 and 3000")
    rng = random.Random(seed)
    now = now or datetime(2025, 3, 14, 12, 0, tzinfo=timezone.utc)
    names = list(_BANK)

    categories = [
        {"_id": _oid("cat", name), "name": name, "slug": name.lower(), "status": "active"} for name in names
    ]
    # A retired duplicate of one category: must merge into the active one.
    categories.append({"_id": _oid("cat", "dup"), "name": "Health and Wellness", "status": "inactive"})
    category_id = {name: _oid("cat", name) for name in names}

    tag_names = sorted({tag for bank in _BANK.values() for tag in bank[4]})
    half = len(tag_names) // 2
    # Two tag collections with different casing, as in production.
    tag_rows = [{"_id": _oid("tag", t), "name": t, "slug": t} for t in tag_names[:half]]
    tags_rows = [{"_id": _oid("tags", t), "name": t.title(), "slug": t} for t in tag_names]
    tag_ids = {t: [_oid("tags", t)] + ([_oid("tag", t)] if t in tag_names[:half] else []) for t in tag_names}

    sources = []
    for name, priority in _OUTLETS:
        row = {
            "_id": _oid("src", name),
            "source_id": name.lower().replace(" ", "_").replace("&", "and"),
            "source_url": f"https://{name.lower().replace(' ', '').replace('&', 'and')}.example",
            "source_icon": None,
            "source_priority": priority,
        }
        row["name:" if name == "Civic Wire" else "name"] = name  # production typo
        sources.append(row)
    english_outlets = [
        s for s, (name, _) in zip(sources, _OUTLETS, strict=False) if name not in _HINDI_OUTLETS
    ]
    hindi_outlets = [s for s, (name, _) in zip(sources, _OUTLETS, strict=False) if name in _HINDI_OUTLETS]

    articles: list[dict[str, Any]] = []
    used_titles: set[str] = set()
    serial = 0

    def add(
        category: str,
        language: str,
        title: str,
        body: str,
        outlet: dict,
        when: datetime,
        summary: str = "",
        **extra: Any,
    ) -> dict:
        nonlocal serial
        serial += 1
        stamp = when.strftime("%Y-%m-%dT%H:%M:%S.") + f"{when.microsecond // 1000:03d}Z"
        row: dict[str, Any] = {
            "_id": {"$oid": _oid("art", serial)},
            "article_id": _oid("aid", serial),
            "title": title,
            "body": body,
            "body_short": summary or body.split(". ")[0].rstrip(".") + ".",
            "thumbnail_url": None,
            "thumbnail_type": None,
            "author_name": "",
            "link": f"{outlet['source_url']}/story/{serial}",
            "date": stamp,
            "language": language,
            "category": [{"$oid": category_id[category]}],
            "tags": [],
            "source": {"$oid": outlet["_id"]},
            "status": "published",
            "updatedAt": {"$date": stamp} if serial % 3 else stamp,
            "sentiment": rng.choice(["positive", "negative", "neutral", "neutral", None]),
        }
        if rng.random() < 0.35:
            tag = rng.choice(_BANK[category][4])
            ref = rng.choice(tag_ids[tag])
            row["tags"] = [ref if rng.random() < 0.6 else {"$oid": ref}]
        if rng.random() < 0.2:
            views = rng.randint(1, 40)
            row.update(
                views=views,
                likes=rng.randint(0, views // 4),
                dislikes=rng.randint(0, 2),
                bookmarks=rng.randint(0, views // 6),
                share=rng.randint(0, 2),
                is_deleted=False,
            )
        row.update(extra)
        articles.append(row)
        return row

    while len(articles) < n_articles:
        category = rng.choice(names)
        subjects, actions, objects, hindi_titles, _ = _BANK[category]
        when = now - timedelta(hours=rng.expovariate(1 / 60.0))
        if rng.random() < 0.3:
            headline = rng.choice(hindi_titles)
            title = f"{rng.choice(_HINDI_PLACES)}: {headline}"
            if title not in used_titles:  # once Hindi headlines run out, fall through to English
                used_titles.add(title)
                person = rng.choice(_HINDI_NAMES)
                body = (
                    f"{headline}। {person} ने बताया कि आगे की जानकारी जल्द दी जाएगी। {rng.choice(_HINDI_FILLER)}"
                )
                add(category, "hindi", title, body, rng.choice(hindi_outlets), when)
                continue
        subject, action, obj = rng.choice(subjects), rng.choice(actions), rng.choice(objects)
        place = rng.choice(_PLACES)
        title = f"{subject} {action} {obj} in {place}"
        if title in used_titles:
            continue
        used_titles.add(title)
        person = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
        lede = rng.choice(_LEDES).format(person=person, place=place, n=rng.randint(3, 400))
        body = f"{title}. {lede}. {rng.choice(_FILLER)}"
        add(
            category,
            "english",
            title,
            body,
            rng.choice(english_outlets),
            when,
            summary=f"{lede}. {body.rsplit('. ', 1)[-1]}",
        )
        # Sometimes other outlets pick the same story up with a reworded headline.
        if rng.random() < 0.12 and len(articles) < n_articles:
            for outlet in rng.sample(english_outlets, rng.randint(1, 2)):
                if len(articles) >= n_articles:
                    break
                echo_title = f"{place}: {subject} {action} {obj}, sources say"
                echo_body = f"{echo_title}. {body}"
                add(
                    category,
                    "english",
                    echo_title,
                    echo_body,
                    outlet,
                    when + timedelta(hours=rng.uniform(0.5, 6)),
                    summary=f"{lede}, according to people briefed on it.",
                )

    # ---- Deliberate mess, on top of the clean articles ----
    sample = articles[0]
    articles.append({**sample, "_id": {"$oid": _oid("art", "dup")}})  # same article_id + link
    articles.append(
        {
            **articles[1],
            "_id": {"$oid": _oid("art", "gone")},
            "article_id": "gone",
            "link": "https://example.invalid/gone",
            "title": "Withdrawn story",
            "is_deleted": True,
        }
    )
    articles.append(
        {
            **articles[2],
            "_id": {"$oid": _oid("art", "draft")},
            "article_id": "draft",
            "link": "https://example.invalid/draft",
            "title": "Unpublished draft",
            "status": "draft",
        }
    )
    legacy = add(
        "Science",
        "english",
        "Observatory logs a legacy date format",
        "A row with a day-first date.",
        english_outlets[0],
        now - timedelta(days=2),
    )
    legacy["date"] = (now - timedelta(days=2)).strftime("%d/%m/%Y %H:%M:%S")
    orphan = add(
        "Science",
        "english",
        "Story with an unknown category reference",
        "Its category id is not in the table.",
        english_outlets[1],
        now - timedelta(days=1),
    )
    orphan["category"] = [{"$oid": "ffffffffffffffffffffffff"}]

    preferences = []
    for u in range(n_users):
        liked = rng.sample(names, rng.randint(1, 4))
        refs = [category_id[name] for name in liked for _ in range(rng.randint(1, 3))]
        row = {
            "_id": _oid("pref", u),
            "user_id": _oid("user", u),
            "language": "hindi" if u % 5 == 0 else "english",
            "article_category": refs,
            "article_tag": [],
            "source": [],
            "updatedAt": "2025-03-01T10:00:00.000000",
        }
        preferences.append(row)
        if u % 9 == 0:  # an older, stale preference document for the same user
            preferences.append(
                {
                    **row,
                    "_id": _oid("pref", u, "old"),
                    "article_category": [category_id[names[0]]],
                    "updatedAt": "2024-01-01T10:00:00.000000",
                }
            )
    preferences.append(
        {
            "_id": _oid("pref", "cold"),
            "user_id": _oid("user", "cold"),
            "language": "english",
            "article_category": [],
            "article_tag": [],
            "source": [],
            "updatedAt": "2025-03-01T10:00:00",
        }
    )

    return {
        "articles": articles,
        "article_categories": categories,
        "article_tags": [],
        "tags": tags_rows,
        "tag": tag_rows,
        "sources": sources,
        "source": [],
        "user_preferences": preferences,
    }
