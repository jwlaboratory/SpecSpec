"""
Deterministic multi-domain prompt generator for the DFlash speculative-decoding
benchmark.

`build_prompts(n_per_category)` returns an ordered dict:
    { category_name: [prompt_str, ...] }   # each list has exactly n_per_category items

Domains covered (the point is BREADTH -- we want to see where a tiny drafter
generalizes to the target and where it doesn't):

  Natural languages : English, Spanish, French, Hindi, German, Chinese,
                      Japanese, Russian, Arabic, Portuguese
  Programming langs : Python, Ruby, JavaScript, C++, SQL, Rust, Go, Bash
  General tasks     : summarization, math_reasoning, question_answering,
                      json_extraction, translation, creative_writing,
                      email_writing, roleplay_chat, logic_reasoning, tabular_data

Prompts are generated combinatorially from per-category template + topic lists,
shuffled with a FIXED seed so every run uses the identical prompt set.
"""
from collections import OrderedDict
import itertools
import random

SEED = 1234


# --------------------------------------------------------------------------- #
# Natural languages: {template with {topic} slot} x {topic}                     #
# Templates and topics are written natively so each category exercises the      #
# model on genuinely in-language text.                                          #
# --------------------------------------------------------------------------- #
LANGUAGES = {
    "lang_english": {
        "templates": [
            "Explain {t} to a curious 12-year-old.",
            "Write a short paragraph about {t}.",
            "List five interesting facts about {t}.",
            "What are the pros and cons of {t}?",
            "Give me practical advice about {t}.",
            "Describe how {t} works, step by step.",
            "Write a persuasive argument in favour of {t}.",
            "Compare {t} with a common alternative.",
            "Tell a very short story involving {t}.",
            "Summarise the main ideas behind {t}.",
        ],
        "topics": [
            "renewable energy", "the human immune system", "compound interest",
            "the printing press", "ocean currents", "machine learning",
            "the water cycle", "ancient Rome", "healthy sleep habits",
            "black holes",
        ],
    },
    "lang_spanish": {
        "templates": [
            "Explica {t} a un niño de doce años.",
            "Escribe un párrafo corto sobre {t}.",
            "Enumera cinco datos interesantes sobre {t}.",
            "¿Cuáles son las ventajas y desventajas de {t}?",
            "Dame consejos prácticos sobre {t}.",
            "Describe paso a paso cómo funciona {t}.",
            "Escribe un argumento a favor de {t}.",
            "Compara {t} con una alternativa común.",
            "Cuenta una historia muy breve sobre {t}.",
            "Resume las ideas principales de {t}.",
        ],
        "topics": [
            "la energía renovable", "el sistema solar", "el interés compuesto",
            "la dieta mediterránea", "las corrientes oceánicas",
            "el aprendizaje automático", "el ciclo del agua",
            "la antigua Roma", "el sueño saludable", "los agujeros negros",
        ],
    },
    "lang_french": {
        "templates": [
            "Explique {t} à un enfant de douze ans.",
            "Écris un court paragraphe sur {t}.",
            "Énumère cinq faits intéressants sur {t}.",
            "Quels sont les avantages et inconvénients de {t} ?",
            "Donne-moi des conseils pratiques sur {t}.",
            "Décris étape par étape comment fonctionne {t}.",
            "Rédige un argument en faveur de {t}.",
            "Compare {t} avec une alternative courante.",
            "Raconte une très brève histoire sur {t}.",
            "Résume les idées principales de {t}.",
        ],
        "topics": [
            "l'énergie renouvelable", "le système solaire", "les intérêts composés",
            "le régime méditerranéen", "les courants océaniques",
            "l'apprentissage automatique", "le cycle de l'eau",
            "la Rome antique", "un bon sommeil", "les trous noirs",
        ],
    },
    "lang_hindi": {
        "templates": [
            "{t} को बारह साल के बच्चे को समझाइए।",
            "{t} के बारे में एक छोटा अनुच्छेद लिखिए।",
            "{t} के बारे में पाँच रोचक तथ्य बताइए।",
            "{t} के फायदे और नुकसान क्या हैं?",
            "{t} के बारे में व्यावहारिक सलाह दीजिए।",
            "{t} कैसे काम करता है, चरण दर चरण बताइए।",
            "{t} के पक्ष में एक तर्क लिखिए।",
            "{t} की तुलना एक सामान्य विकल्प से कीजिए।",
            "{t} पर एक बहुत छोटी कहानी सुनाइए।",
            "{t} के मुख्य विचारों का सारांश दीजिए।",
        ],
        "topics": [
            "नवीकरणीय ऊर्जा", "सौर मंडल", "चक्रवृद्धि ब्याज",
            "संतुलित आहार", "महासागरीय धाराएँ", "मशीन लर्निंग",
            "जल चक्र", "प्राचीन भारत", "अच्छी नींद", "ब्लैक होल",
        ],
    },
    "lang_german": {
        "templates": [
            "Erkläre {t} einem zwölfjährigen Kind.",
            "Schreibe einen kurzen Absatz über {t}.",
            "Nenne fünf interessante Fakten über {t}.",
            "Was sind die Vor- und Nachteile von {t}?",
            "Gib mir praktische Ratschläge zu {t}.",
            "Beschreibe Schritt für Schritt, wie {t} funktioniert.",
            "Schreibe ein Argument für {t}.",
            "Vergleiche {t} mit einer gängigen Alternative.",
            "Erzähle eine sehr kurze Geschichte über {t}.",
            "Fasse die wichtigsten Ideen von {t} zusammen.",
        ],
        "topics": [
            "erneuerbare Energie", "das Sonnensystem", "der Zinseszins",
            "gesunde Ernährung", "Meeresströmungen", "maschinelles Lernen",
            "der Wasserkreislauf", "das antike Rom", "gesunder Schlaf",
            "schwarze Löcher",
        ],
    },
    "lang_chinese": {
        "templates": [
            "请向一个十二岁的孩子解释{t}。",
            "写一段关于{t}的短文。",
            "列出关于{t}的五个有趣事实。",
            "{t}有哪些优点和缺点？",
            "给我一些关于{t}的实用建议。",
            "一步一步描述{t}是如何运作的。",
            "写一段支持{t}的论述。",
            "把{t}与一个常见的替代方案进行比较。",
            "讲一个关于{t}的很短的故事。",
            "总结{t}的主要思想。",
        ],
        "topics": [
            "可再生能源", "太阳系", "复利", "健康饮食", "洋流",
            "机器学习", "水循环", "古罗马", "健康的睡眠", "黑洞",
        ],
    },
    "lang_japanese": {
        "templates": [
            "{t}を12歳の子供に説明してください。",
            "{t}について短い段落を書いてください。",
            "{t}に関する面白い事実を五つ挙げてください。",
            "{t}の長所と短所は何ですか？",
            "{t}についての実用的なアドバイスをください。",
            "{t}の仕組みを順を追って説明してください。",
            "{t}を支持する主張を書いてください。",
            "{t}を一般的な代替案と比較してください。",
            "{t}にまつわるとても短い物語を語ってください。",
            "{t}の主な考え方を要約してください。",
        ],
        "topics": [
            "再生可能エネルギー", "太陽系", "複利", "健康的な食事", "海流",
            "機械学習", "水の循環", "古代ローマ", "健康的な睡眠", "ブラックホール",
        ],
    },
    "lang_russian": {
        "templates": [
            "Объясни {t} двенадцатилетнему ребёнку.",
            "Напиши короткий абзац о {t}.",
            "Перечисли пять интересных фактов о {t}.",
            "Каковы плюсы и минусы {t}?",
            "Дай практические советы о {t}.",
            "Опиши шаг за шагом, как работает {t}.",
            "Напиши аргумент в пользу {t}.",
            "Сравни {t} с распространённой альтернативой.",
            "Расскажи очень короткую историю о {t}.",
            "Кратко изложи основные идеи {t}.",
        ],
        "topics": [
            "возобновляемая энергия", "Солнечная система", "сложные проценты",
            "здоровое питание", "океанские течения", "машинное обучение",
            "круговорот воды", "древний Рим", "здоровый сон", "чёрные дыры",
        ],
    },
    "lang_arabic": {
        "templates": [
            "اشرح {t} لطفل عمره اثنا عشر عامًا.",
            "اكتب فقرة قصيرة عن {t}.",
            "اذكر خمس حقائق مثيرة للاهتمام عن {t}.",
            "ما هي مزايا وعيوب {t}؟",
            "قدّم لي نصائح عملية حول {t}.",
            "صف خطوة بخطوة كيف يعمل {t}.",
            "اكتب حجة تؤيد {t}.",
            "قارن {t} ببديل شائع.",
            "احكِ قصة قصيرة جدًا عن {t}.",
            "لخّص الأفكار الرئيسية لـ {t}.",
        ],
        "topics": [
            "الطاقة المتجددة", "النظام الشمسي", "الفائدة المركبة",
            "الغذاء الصحي", "التيارات المحيطية", "تعلّم الآلة",
            "دورة الماء", "روما القديمة", "النوم الصحي", "الثقوب السوداء",
        ],
    },
    "lang_portuguese": {
        "templates": [
            "Explique {t} para uma criança de doze anos.",
            "Escreva um parágrafo curto sobre {t}.",
            "Liste cinco fatos interessantes sobre {t}.",
            "Quais são as vantagens e desvantagens de {t}?",
            "Dê-me conselhos práticos sobre {t}.",
            "Descreva passo a passo como {t} funciona.",
            "Escreva um argumento a favor de {t}.",
            "Compare {t} com uma alternativa comum.",
            "Conte uma história bem curta sobre {t}.",
            "Resuma as ideias principais de {t}.",
        ],
        "topics": [
            "a energia renovável", "o sistema solar", "os juros compostos",
            "a alimentação saudável", "as correntes oceânicas",
            "o aprendizado de máquina", "o ciclo da água",
            "a Roma antiga", "o sono saudável", "os buracos negros",
        ],
    },
}


# --------------------------------------------------------------------------- #
# Programming languages: shared task list, per-language wrapper                  #
# --------------------------------------------------------------------------- #
CODE_TASKS = [
    "reverses a linked list", "checks whether a string is a palindrome",
    "computes the nth Fibonacci number iteratively", "sorts a list using quicksort",
    "finds the greatest common divisor of two integers",
    "counts the frequency of each word in a paragraph",
    "merges two sorted arrays", "detects a cycle in a directed graph",
    "implements binary search", "flattens an arbitrarily nested list",
    "converts a Roman numeral to an integer", "validates an email address",
    "computes the factorial of a number", "removes duplicates from a list while keeping order",
    "finds the longest common prefix of a list of strings",
    "implements a simple LRU cache", "parses a CSV line into fields",
    "computes the moving average of a numeric stream",
    "returns the prime numbers below n using a sieve",
    "rotates a matrix 90 degrees clockwise",
    "checks if two strings are anagrams", "implements a stack using two queues",
    "finds the second largest element in an array",
    "computes the edit distance between two strings",
    "groups anagrams together from a list of words",
    "implements run-length encoding and decoding",
    "finds all pairs in an array that sum to a target",
    "evaluates a string containing a simple arithmetic expression",
    "computes the depth of a binary tree",
    "implements a debounce wrapper for a function",
    "converts an integer to its binary representation",
    "finds the majority element in an array",
    "implements insertion sort", "counts set bits in an integer",
    "checks whether parentheses in a string are balanced",
    "computes the intersection of two lists",
    "implements a basic retry-with-backoff helper",
    "finds the shortest path in an unweighted grid using BFS",
    "computes the transpose of a matrix",
    "implements a simple exponential moving average",
    "returns the k most frequent elements in a list",
    "implements a min-heap with push and pop",
    "checks if a number is a perfect square without using sqrt",
    "generates all permutations of a short string",
    "implements a simple rate limiter",
    "computes the Hamming distance between two equal-length strings",
    "finds the missing number in a range of 1..n",
    "implements a trie with insert and search",
    "computes the running maximum over a sliding window",
    "converts snake_case to camelCase",
]

CODE_LANGS = {
    "code_python": "Python",
    "code_ruby": "Ruby",
    "code_javascript": "JavaScript",
    "code_cpp": "C++",
    "code_sql": "SQL",
    "code_rust": "Rust",
    "code_go": "Go",
    "code_bash": "Bash",
}

CODE_TEMPLATES = [
    "Write a {lang} function that {task}. Include a short comment explaining the approach.",
    "Implement, in {lang}, code that {task}. Add a couple of example calls.",
    "Write clean, idiomatic {lang} that {task}.",
    "In {lang}, write a function that {task} and explain its time complexity.",
]

# SQL doesn't map onto most algorithmic tasks; give it its own realistic task set.
SQL_TASKS = [
    "returns the top 5 customers by total spend",
    "finds all orders placed in the last 30 days",
    "computes monthly revenue grouped by product category",
    "lists employees who earn more than their manager",
    "finds duplicate email addresses in a users table",
    "computes a 7-day rolling average of daily sales",
    "returns the second highest salary in each department",
    "joins orders, customers, and products into one report",
    "counts active users per day for the past week",
    "finds products that have never been ordered",
    "computes the churn rate between two months",
    "ranks students by grade within each class",
    "returns customers who ordered every product in a category",
    "finds the median order value per region",
    "computes year-over-year growth per store",
    "lists the most recent order for each customer",
    "finds gaps in a sequence of invoice numbers",
    "computes the percentage of orders that were refunded",
    "returns sessions longer than the average session length",
    "pivots monthly sales into one row per product",
]


# --------------------------------------------------------------------------- #
# General task domains                                                          #
# --------------------------------------------------------------------------- #
_SUMMARY_TOPICS = [
    "urban beekeeping", "the history of coffee", "coral reef restoration",
    "the invention of the transistor", "migratory bird navigation",
    "the economics of public transit", "how vaccines are developed",
    "the rise of remote work", "glacier retreat in the Alps",
    "the physics of suspension bridges", "the domestication of dogs",
    "how lithium batteries are recycled", "the spread of the printing press",
    "tidal energy generation", "the psychology of habit formation",
    "the restoration of wetlands", "how GPS satellites keep time",
    "the global supply chain for cocoa", "volcanic soil and agriculture",
    "the history of the marathon",
]

_QA_QUESTIONS = [
    "Why is the sky blue?", "How do noise-cancelling headphones work?",
    "What causes the seasons?", "Why does bread rise?",
    "How do airplanes stay in the air?", "What is the difference between weather and climate?",
    "Why do we get jet lag?", "How does a refrigerator keep food cold?",
    "What makes a rainbow appear?", "Why do onions make you cry?",
    "How does the internet route data?", "What is compound interest?",
    "Why do metals feel colder than wood?", "How do bees make honey?",
    "What is the greenhouse effect?", "Why does ice float on water?",
    "How do vaccines train the immune system?", "What causes tides?",
    "Why do stars twinkle?", "How does a microwave heat food?",
]

_JSON_SENTENCES = [
    "Maria Lopez, 34, is a data scientist at Acme Corp in Berlin and earns 92000 euros.",
    "The order #A1029 for 3 blue mugs shipped on 2024-05-02 to Toronto, total 45.00 USD.",
    "Flight LH431 departs Munich at 09:15 and arrives in Boston at 12:40, gate B22.",
    "James, a 27-year-old nurse from Leeds, adopted a 2-year-old beagle named Rex.",
    "The book 'Deep Roots' by A. Okafor has 312 pages and was published in 2019.",
    "Invoice 5567: 2 keyboards at 30 each and 1 monitor at 210, due 2024-06-30.",
    "Dr. Chen scheduled a cardiology appointment for patient #8842 on Friday at 3pm.",
    "The cafe on 5th Street opens at 7am, closes at 6pm, and seats 40 people.",
    "Team Falcon scored 3 goals, Team Hawk scored 1, match played on 2023-11-12.",
    "Sensor A reported 23.4C and 61% humidity at 14:05 in greenhouse 2.",
    "Customer 771 returned 2 items worth 58.50 GBP and requested a refund.",
    "The conference runs March 3-5 in Austin with 1200 attendees and 40 speakers.",
    "A 500g bag of dark-roast coffee costs 12.99 and ships in 2 business days.",
    "Employee E-204, Priya Nair, joined the design team on 2021-08-16 in Pune.",
    "The recipe needs 250g flour, 2 eggs, 100ml milk, and bakes for 25 minutes.",
    "Ticket #INC-9931 (priority high) was opened at 08:12 and closed at 11:47.",
    "The apartment at 14 Elm Road has 2 bedrooms, 1 bath, and rents for 1400/month.",
    "Bus route 12 leaves Central at 08:00 and reaches the harbour in 35 minutes.",
    "Order line: 4 units of SKU-7781 at 9.25 each, warehouse W3, backordered.",
    "Runner #452, age 41, finished the 10k in 48 minutes and 30 seconds.",
]

_TRANSLATE_SENTENCES = [
    "The weather is lovely today, so let's go for a walk in the park.",
    "Could you please tell me how to get to the nearest train station?",
    "She has been studying medicine for six years and graduates next spring.",
    "We should book the tickets early to get a better price.",
    "The new policy will take effect at the beginning of next month.",
    "I really enjoyed the film, but the ending was a little confusing.",
    "Please remember to water the plants while I am away.",
    "The company plans to open three new offices across Europe.",
    "He apologised for being late and promised it would not happen again.",
    "Reading before bed helps me fall asleep more easily.",
    "The museum is free on Sundays and stays open until eight.",
    "They are renovating the old bridge to make it safer for cyclists.",
    "My grandmother taught me how to make bread from scratch.",
    "The meeting has been moved to Thursday afternoon at three.",
    "This restaurant is famous for its fresh seafood and friendly staff.",
    "We watched the sunset from the top of the hill and it was beautiful.",
    "The children built a sandcastle and then the waves washed it away.",
    "I need to charge my phone before we leave for the airport.",
    "The teacher explained the problem again until everyone understood.",
    "A balanced diet and regular exercise keep you healthy.",
]
_TRANSLATE_TARGETS = ["Spanish", "French", "German", "Hindi", "Japanese"]

_CREATIVE = [
    "Write a four-line poem about a city waking up at dawn.",
    "Write the opening paragraph of a mystery set in a lighthouse.",
    "Write a short bedtime story about a shy dragon who loves gardening.",
    "Write a limerick about a cat who thinks it is a chef.",
    "Write a 100-word sci-fi flash story about the last library on Earth.",
    "Write a dialogue between the moon and a passing comet.",
    "Write a product description for a mug that keeps coffee warm forever.",
    "Write a haiku about the first snow of winter.",
    "Write a short fable whose moral is about patience.",
    "Write a diary entry from the point of view of a houseplant.",
    "Write a toast for a friend opening their first bakery.",
    "Write a spooky campfire story in exactly five sentences.",
    "Write a poem where each line starts with the next letter of the word OCEAN.",
    "Write a short scene where two strangers share an umbrella.",
    "Write an inspiring speech for a robot graduating from robot school.",
    "Write a nursery rhyme about brushing your teeth.",
    "Write a letter from a future self to a nervous first-year student.",
    "Write a short story that begins with 'The map was wrong.'",
    "Write a playful advertisement for rain.",
    "Write a monologue for a very dramatic teapot.",
]

_EMAIL = [
    "Write a polite email asking your manager to work from home on Fridays.",
    "Write an email to a client apologising for a shipping delay.",
    "Write a follow-up email after a job interview for a marketing role.",
    "Write an email inviting your team to a project kickoff meeting.",
    "Write an email requesting a refund for a defective laptop.",
    "Write an email introducing yourself to a new team as their manager.",
    "Write an email declining a meeting invitation politely.",
    "Write an email thanking a mentor for their guidance this year.",
    "Write an email to a landlord reporting a broken heater.",
    "Write an email asking a professor for a deadline extension.",
    "Write an email announcing a new feature to your customers.",
    "Write an email to reschedule a dentist appointment.",
    "Write an email negotiating a lower price with a supplier.",
    "Write an email welcoming a new subscriber to a newsletter.",
    "Write an email to a colleague handing off a project before vacation.",
    "Write an email reminding attendees to submit their slides.",
    "Write an email congratulating a coworker on a promotion.",
    "Write an email asking for feedback on a draft proposal.",
    "Write an email cancelling a subscription and explaining why.",
    "Write an email to a charity offering to volunteer on weekends.",
]

_ROLEPLAY = [
    "You are a friendly museum guide. A visitor asks what to see first.",
    "You are a calm 911 dispatcher. Someone reports a kitchen fire.",
    "You are a medieval blacksmith. A knight wants a lighter sword.",
    "You are a patient math tutor. A student is stuck on fractions.",
    "You are a travel agent. A couple wants a quiet beach holiday.",
    "You are a wise old tree. A child asks why leaves fall.",
    "You are a startup mentor. A founder pitches a food-delivery app.",
    "You are a barista who is also a poet. A regular orders their usual.",
    "You are a ship's captain in a storm. Reassure the passengers.",
    "You are a librarian. Recommend a book for someone who hates reading.",
    "You are a friendly robot chef. Suggest a dinner using only leftovers.",
    "You are a fitness coach. A beginner asks how to start running.",
    "You are a detective. Explain your first three steps at a crime scene.",
    "You are a gardener. Advise someone whose tomatoes keep dying.",
    "You are an air traffic controller. Guide a nervous student pilot.",
    "You are a career counselor. Help a teacher who wants a new field.",
    "You are a museum dinosaur come to life. Greet the schoolchildren.",
    "You are a hotel concierge. A guest wants a surprise anniversary plan.",
    "You are a friendly alien. Ask three questions about human breakfast.",
    "You are a veteran mountaineer. Warn a hiker about the weather ahead.",
]

_LOGIC = [
    "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Explain.",
    "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
    "Three people check into a hotel and the bellboy pockets some money. Explain the missing dollar puzzle.",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    "You have two ropes that each burn in 60 minutes unevenly. Measure 45 minutes.",
    "A farmer must cross a river with a fox, a chicken, and grain. Plan the crossings.",
    "In a race you pass the person in second place. What place are you in now?",
    "Some months have 30 days, some 31. How many have 28 days?",
    "A snail climbs 3 feet a day and slips 2 feet at night in a 10-foot well. When does it escape?",
    "If five cats catch five mice in five minutes, how many cats catch 100 mice in 100 minutes?",
    "Two trains 100 miles apart approach at 50 mph each. A bird flies between them at 75 mph. How far does the bird fly?",
    "You have 8 balls, one heavier. Find it in two weighings on a balance scale.",
    "A clock strikes 6 in 5 seconds. How long to strike 12?",
    "If today is two days before the day after tomorrow's yesterday, what day is it? Explain your reasoning.",
    "There are three boxes labelled wrongly: apples, oranges, both. Identify all with one pick.",
    "A man looks at a portrait and says 'brothers and sisters I have none, but that man's father is my father's son.' Who is it?",
    "You need exactly 4 litres using a 3-litre and a 5-litre jug. How?",
    "Twelve coins, one is fake and differently weighted. Find it in three weighings.",
    "If half of 5 is 3, what is a third of 10 under the same rule? Explain the pattern.",
    "A rope ladder hangs off a boat with rungs 1 foot apart. The tide rises 2 feet. How many rungs are underwater? Explain.",
]

_TABULAR = [
    "Given the table:\nName | Sales | Region\nAmy | 120 | North\nBob | 90 | South\nCara | 150 | North\nWho had the most sales, and what is the total for the North region?",
    "Given the table:\nMonth | Users\nJan | 1000\nFeb | 1500\nMar | 1200\nWhat was the percentage change from January to February?",
    "Given the table:\nProduct | Price | Qty\nPen | 2 | 10\nPad | 5 | 4\nInk | 8 | 3\nCompute the total revenue and the most expensive line item.",
    "Given the table:\nCity | Temp\nOslo | -3\nCairo | 28\nLima | 19\nList the cities from coldest to warmest.",
    "Given the table:\nTeam | Wins | Losses\nRed | 8 | 2\nBlue | 5 | 5\nGold | 3 | 7\nWhich team has the best win rate, and what is it as a percentage?",
    "Given the table:\nEmployee | Hours | Rate\nLee | 40 | 25\nMax | 35 | 30\nCompute each person's pay and the total payroll.",
    "Given the table:\nQuarter | Revenue | Cost\nQ1 | 500 | 300\nQ2 | 700 | 450\nCompute profit per quarter and total profit.",
    "Given the table:\nStudent | Math | Science\nRia | 88 | 92\nSam | 76 | 80\nWho has the higher average, and what are the two averages?",
    "Given the table:\nItem | Stock | Reorder\nA | 12 | 20\nB | 30 | 15\nC | 5 | 10\nWhich items are below their reorder level?",
    "Given the table:\nDay | Steps\nMon | 8000\nTue | 12000\nWed | 6000\nWhat is the average daily step count, and which day was highest?",
    "Given the table:\nRoute | Distance | Fuel\nX | 120 | 10\nY | 90 | 6\nWhich route is more fuel efficient in km per litre?",
    "Given the table:\nName | Age | City\nIvy | 30 | Rome\nJon | 45 | Rome\nKai | 22 | Oslo\nWhat is the average age of people in Rome?",
    "Given the table:\nWeek | Signups | Churn\n1 | 200 | 20\n2 | 240 | 30\nCompute net new users each week and the total.",
    "Given the table:\nModel | Price | Rating\nA | 300 | 4.5\nB | 250 | 4.0\nC | 400 | 4.8\nWhich model has the best rating-per-dollar?",
    "Given the table:\nRegion | Q1 | Q2\nEast | 100 | 130\nWest | 80 | 60\nWhich region grew and by what percentage?",
    "Given the table:\nFood | Calories | Grams\nRice | 130 | 100\nNuts | 600 | 100\nWhich food is more calorie-dense per gram?",
    "Given the table:\nName | Score\nA | 45\nB | 78\nC | 62\nD | 90\nWhat is the median score?",
    "Given the table:\nStore | Jan | Feb | Mar\nS1 | 10 | 12 | 15\nS2 | 20 | 18 | 22\nWhich store had the most total sales over the quarter?",
    "Given the table:\nTask | Est | Actual\nT1 | 5 | 7\nT2 | 8 | 6\nWhich tasks went over estimate, and by how much in total?",
    "Given the table:\nPlan | Users | Price\nFree | 500 | 0\nPro | 120 | 15\nCompute total monthly revenue across plans.",
]


def _fill(templates, topics, n):
    """Cross product of templates x topics, shuffled deterministically, sliced to n."""
    combos = [t.replace("{t}", topic) for t, topic in itertools.product(templates, topics)]
    rng = random.Random(SEED)
    rng.shuffle(combos)
    # Ensure we can reach n even if product < n by cycling (kept unique-ish by index).
    if len(combos) < n:
        combos = (combos * ((n // len(combos)) + 1))
    return combos[:n]


def _fill_code(lang, n):
    tasks = SQL_TASKS if lang == "SQL" else CODE_TASKS
    combos = [tpl.replace("{lang}", lang).replace("{task}", task)
              for tpl, task in itertools.product(CODE_TEMPLATES, tasks)]
    rng = random.Random(SEED)
    rng.shuffle(combos)
    if len(combos) < n:
        combos = combos * ((n // len(combos)) + 1)
    return combos[:n]


def _cycle_to_n(seed_prompts, n, prefix_variations):
    """Turn a short seed list into n prompts by pairing each seed with a set of
    lightweight instruction variations, deterministically shuffled."""
    combos = [f"{var}{s}" if var else s
              for var, s in itertools.product(prefix_variations, seed_prompts)]
    rng = random.Random(SEED)
    rng.shuffle(combos)
    if len(combos) < n:
        combos = combos * ((n // len(combos)) + 1)
    return combos[:n]


def build_prompts(n_per_category=100):
    cats = OrderedDict()

    # Natural languages
    for name, spec in LANGUAGES.items():
        cats[name] = _fill(spec["templates"], spec["topics"], n_per_category)

    # Programming languages
    for name, lang in CODE_LANGS.items():
        cats[name] = _fill_code(lang, n_per_category)

    # General tasks -------------------------------------------------------- #
    cats["task_summarization"] = _cycle_to_n(
        [f"Summarise the key points about {t} in three sentences." for t in _SUMMARY_TOPICS],
        n_per_category,
        ["", "In plain language, ", "For a busy reader, ", "Concisely, ", "Without jargon, "],
    )
    cats["task_question_answering"] = _cycle_to_n(
        _QA_QUESTIONS, n_per_category,
        ["", "Explain simply: ", "In detail, answer: ", "Briefly: ", "For a beginner: "],
    )
    cats["task_json_extraction"] = _cycle_to_n(
        [f"Extract the structured fields from this sentence as JSON:\n{s}" for s in _JSON_SENTENCES],
        n_per_category,
        ["", "Return only valid JSON. ", "Use snake_case keys. ", "Include a 'type' field. ", "Be precise. "],
    )
    cats["task_translation"] = _cycle_to_n(
        [f"Translate the following into {tgt}:\n\"{s}\""
         for tgt in _TRANSLATE_TARGETS for s in _TRANSLATE_SENTENCES],
        n_per_category, [""],
    )
    cats["task_creative_writing"] = _cycle_to_n(
        _CREATIVE, n_per_category,
        ["", "Be imaginative. ", "Keep it warm and vivid. ", "Use simple words. ", "Make it playful. "],
    )
    cats["task_email_writing"] = _cycle_to_n(
        _EMAIL, n_per_category,
        ["", "Keep it under 120 words. ", "Use a professional tone. ", "Be warm but concise. ", "Include a clear subject line. "],
    )
    cats["task_roleplay_chat"] = _cycle_to_n(
        _ROLEPLAY, n_per_category,
        ["", "Stay fully in character. ", "Reply in two short paragraphs. ", "Be helpful and vivid. ", "Keep it friendly. "],
    )
    cats["task_math_reasoning"] = _cycle_to_n(
        [
            "A train travels 240 km in 3 hours. What is its average speed, and how far in 5 hours?",
            "A shirt costs $40 after a 20% discount. What was the original price?",
            "If 3 pens cost $7.50, how much do 11 pens cost?",
            "A rectangle has area 48 and width 6. What is its perimeter?",
            "You invest $1000 at 5% simple interest for 3 years. What is the total?",
            "A recipe for 4 needs 300g flour. How much for 7 people?",
            "The angles of a triangle are in ratio 2:3:4. Find each angle.",
            "A car uses 6 litres per 100 km. How much fuel for a 450 km trip?",
            "If x + 2y = 10 and x = 4, find y and then 3x - y.",
            "A tank fills in 12 minutes with one pipe and 18 with another. How long together?",
            "A number increased by 15% becomes 92. What was the number?",
            "Two dice are rolled. What is the probability the sum is 7?",
            "A ladder 13 m long leans against a wall, base 5 m out. How high does it reach?",
            "Solve for x: 2(x - 3) = 3x + 4.",
            "A population grows from 800 to 1000 in a year. What is the growth rate?",
            "A pizza is cut into 8 slices; 3 are eaten. What percentage remains?",
            "The average of five numbers is 20. Four of them are 15, 18, 22, 25. Find the fifth.",
            "A cyclist covers 2/3 of a route in 40 minutes. How long for the whole route at that pace?",
            "Convert 3/8 to a decimal and a percentage.",
            "A cube has volume 27 cm³. What is the length of one edge and its surface area?",
        ],
        n_per_category,
        ["Solve step by step. ", "Show your reasoning: ", "Work it out carefully: ", "Explain each step. ", "Reason it through: "],
    )
    cats["task_logic_reasoning"] = _cycle_to_n(
        _LOGIC, n_per_category,
        ["", "Think step by step. ", "Explain your reasoning: ", "Reason carefully: ", "Justify your answer: "],
    )
    cats["task_tabular_data"] = _cycle_to_n(
        _TABULAR, n_per_category,
        ["", "Show your working. ", "Answer precisely. ", "Explain briefly. ", "Give the numbers: "],
    )

    # Sanity: every category has exactly n_per_category prompts.
    for name, plist in cats.items():
        assert len(plist) == n_per_category, f"{name} has {len(plist)} != {n_per_category}"
    return cats


CATEGORY_GROUPS = {
    "languages": [k for k in LANGUAGES],
    "coding": [k for k in CODE_LANGS],
    "tasks": [
        "task_summarization", "task_question_answering", "task_json_extraction",
        "task_translation", "task_creative_writing", "task_email_writing",
        "task_roleplay_chat", "task_math_reasoning", "task_logic_reasoning",
        "task_tabular_data",
    ],
}


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    data = build_prompts(n)
    print(f"{len(data)} categories x {n} prompts = {len(data) * n} total prompts\n")
    for cat, plist in data.items():
        print(f"  {cat:28s} e.g.  {plist[0][:70]!r}")
