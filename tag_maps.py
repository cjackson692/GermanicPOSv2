
PENN_BASE = {
    **{t: "VB" for t in
       "VB VBI VBPH VBPI VBPS VBP VBDI VBDS VBD VAG VBN VAN "
       "BE BEI BEPH BEPI BEPS BEP BEDI BEDS BED BAG BEN BAN "
       "HV HVI HVPI HVPS HVP HVDI HVDS HVD HAG HVN HAN "
       "AX AXI AXPI AXPS AXP AXDI AXDS AXD AXG AXN "
       "MD MDI MDPI MDPS MDP MDDI MDDS MDD MDN "
       "DO DOI DOPI DOPS DOP DODI DODS DOD DAG DON DAN "
       "RD RDI RDPI RDPS RDP RDDI RDDS RDD RAG RDN RAN "
       "VBG BAG HAG DAG VN VNI VG VGI".split()},
    "TO": "TO",                     
    "N": "N", "NS": "N", "NR": "N", "NPR": "N", "NPRS": "N", "MAN": "N",
    "PRO": "PRO", "PROS": "PRO$", "PRO$": "PRO$",
    "D": "D", "Q": "Q", "QR": "Q", "QS": "Q",
    "ADJ": "ADJ", "ADJR": "ADJ", "ADJS": "ADJ",
    "ADV": "ADV", "ADVR": "ADV", "ADVS": "ADV",
    "P": "AP", "C": "C", "CONJ": "CONJ", "NEG": "NEG",
    "NUM": "NUM", "INTJ": "INTJ",
    "FP": "PART", "RP": "PART", "RPX": "PART",
    "WPRO": "WH", "WADV": "WH", "WADJ": "WH", "WQ": "WH", "WD": "WH",
    "ONE": "ONE", "ONES": "ONE", "OTHER": "OTHER", "OTHERS": "OTHER",
    "SUCH": "SUCH", "ALSO": "ALSO", "ES": "ES",
    "X": "XX", "XX": "XX", "FW": "XX", "UNKNOWN": "XX", "MISSING": "XX",
}
PENN_PUNCT = {",", ".", "'", '"', ";", ":", "$"}

GOTHIC = {
    "Verb": "VB", "Participle": "VB",
    "Noun": "N", "Proper Noun": "N", "Name": "N",
    "Pronoun": "PRO", "Preposition": "AP", "Adverb": "ADV",
    "Conjunction": "CONJ", "Adjective": "ADJ", "Numeral": "NUM",
    "Particle": "PART", "Interjection": "INTJ",
    "Unassigned": "XX",            
    "Foreign word": "XX", "nan": "XX",
    "Punctuation": "PUNCT",
}

OHG_POS = {
    **{t: "VB" for t in
       "VV VVFIN VVINF VVINFS VVIMP VVPP VVPS VVPSA VVPSS "
       "VA VAFIN VAINF VAINFS VAIMP VAPP VAPS "
       "VM VMFIN VMINF VMPS VVPSD VVPPD VVPPA VVPSN VVPPS VVPPN VAPSD VAPPD".split()},
    "N": "N", "NA": "N", "NE": "N", "NEO": "N", 
    "ADJS": "ADJS", "ADJOS": "ADJOS", "CARDS": "CARDS",
    "ADJ": "ADJ", "ADJA": "ADJ", "ADJD": "ADJ", "ADJN": "ADJ", "ADJO": "ADJ",
    "PPER": "PRO", "PRF": "PRO", "PI": "PRO", "PIS": "PRO", "PINEG": "PRO",
    **{t: "D" for t in "DD DDA DDS DDSREL DDN DPOS DPOSA DPOSS DPOSN DI DIA DIS DIN DINEG DREL".split()},
    "AP": "AP", "APPR": "AP", "APPO": "AP", "APZR": "AP",
    "ADV": "ADV", "AVD": "ADV", "AVREL": "ADV", "AVW": "WH", "AVG": "ADV",
    "ADVNEG": "ADV",     # negated adverb (parallels YCOE ADV for "never"-type)
    "KO": "CONJ",      
    "KON": "CONJ",     
    "KOUS": "C",       
    "KOUI": "C",       
    "KOU": "C",        
    "KOKOM": "KOKOM",  
    "PTK": "PART", "PTKA": "PART", "PTKANT": "PART", "PTKVZ": "PART",
    "PTKNEG": "NEG",   
    "PTKZU": "TO",     
    "PTKREL": "PART", "PTKW": "PART", "PTKIJ": "INTJ",
    "PW": "WH", "PWG": "WH", "PWS": "WH", "PWA": "WH", "DW": "WH", "PWAV": "WH",
    "PWAVREL": "WH", "PWREL": "WH", "DWSREL": "WH", "DWA": "WH", "DWS": "WH",
    "CARD": "NUM", "ITJ": "INTJ",
    "$,": "PUNCT", "$.": "PUNCT", "$(": "PUNCT", "$_": "PUNCT",
}

FINAL_TAG_DESCRIPTIONS = {
    "VB": "verb (all finite/non-finite, auxiliary, modal)",
    "N": "noun (common, proper; OHG incl. nominalized adjectives per config)",
    "PRO": "personal/indefinite/reflexive pronoun", "PRO$": "possessive pronoun (Penn corpora)",
    "D": "determiner (articles, demonstratives incl. relative-use)",
    "Q": "quantifier (Penn corpora)", "ADJ": "adjective", "ADV": "adverb",
    "AP": "adposition", "CONJ": "coordinating conjunction",
    "C": "complementizer / subordinating conjunction",
    "NEG": "negation marker", "NUM": "numeral (cardinal)",
    "PART": "particle (verbal, focus, answer...)", "TO": "infinitive marker (v2)",
    "WH": "interrogative/wh word", "INTJ": "interjection",
    "ONE": "IcePaHC 'einn' (v2: kept distinct)", "OTHER": "IcePaHC 'annar' (v2: kept distinct)",
    "ADJS": "OHG nominalized adjective (source label preserved; non-comparable)",
    "ADJOS": "OHG nominalized ordinal (source label preserved)",
    "CARDS": "OHG nominalized cardinal (source label preserved)",
    "KOKOM": "OHG comparative particle (between C and CONJ; source label preserved)",
    "SUCH": "IcePaHC 'slíkur'", "ALSO": "IcePaHC 'einnig/líka'", "ES": "IcePaHC expletive",
    "PUNCT": "punctuation (masked)", "XX": "unknown/foreign/unresolvable (masked)",
    "start": "sequence boundary (masked)", "stop": "sequence boundary (masked)",
}
