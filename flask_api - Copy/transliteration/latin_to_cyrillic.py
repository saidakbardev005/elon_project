def latin_to_cyrillic(text: str) -> str:
    """
    Lotincha matnni kiril alifbosiga transliteratsiya qiladi.
    Qoʻshma harflar (masalan, "sh", "ch", "ng", "o‘", "g‘", "yo", "yu", "ya", "ye")
    alohida roʻyxat shaklida avvaliga ishlov beriladi, soʻng bitta harflar.
    Bosh va oxiridagi boʻshliqlar strip() qilinadi.
    """

    # 1) Avval bosh va oxiridagi bo'shliqlarni olib tashlaymiz
    text = text.strip()

    # 2) Transliteratsiya uchun map yaratamiz
    replace_pairs = {
        # Qo'shma birliklar (uzunroq belgilar birinchi kelishi muhim)
        "yo": "ё",
        "yu": "ю",
        "ya": "я",
        "ye": "е",
        "o‘": "ў",
        "g‘": "ғ",
        "sh": "ш",
        "ch": "ч",
        "ng": "нг",
        "o'": "ў",
        "g'": "ғ",

        # Bitta harfli mapping
        "a": "а", "b": "б", "d": "д", "e": "э", "f": "ф",
        "g": "г", "h": "ҳ", "i": "и", "j": "ж", "k": "к",
        "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
        "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у",
        "v": "в", "x": "х", "y": "й", "z": "з",

        # Turli apostrof belgilarini yo‘q qilish (bo‘sh simvolga almashtiramiz)
        "’": "",  # o‘rta apostrof
        "'": "",  # bitta apostrof
        "ʻ": "",  # sotuplam apostrof (UTF-8 ning boshqa varianti)
        "`": "",  # og‘izch call (backtick)
    }

    # 3) Katta harflar uchun ham mapping yaratamiz (upper/lower formatlarini saqlab)
    #    Masalan: "sh" -> "ш", shuning uchun "Sh" -> "Ш", "SH" -> "Ш" kabi bo‘lishi lozim.
    #    Biz original mapping bo‘yicha har bir kalitni upper() ga o‘tkazamiz va qiymatni upper() qiliamiza.
    for latin, cyrillic in list(replace_pairs.items()):
        replace_pairs[latin.upper()] = cyrillic.upper()

    # 4) Qo‘shma birliklar ustuvorligi uchun uzunroq kalitlardan boshlab almashtirishni tartibga solamiz.
    #    sorted_pairs – kalit uzunligi bo‘yicha kamayish tartibida sort qilingan ro‘yxat.
    sorted_pairs = sorted(replace_pairs.items(), key=lambda x: -len(x[0]))

    # 5) Asosiy transliteratsiya jarayoni
    result = text
    for latin, cyrillic in sorted_pairs:
        result = result.replace(latin, cyrillic)

    return result
