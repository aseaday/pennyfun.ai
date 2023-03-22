import json
dataset = []
zh = open("/home/aseaday/nmt-corpus/webnovel/train.raw.zh").readlines()
en = open("/home/aseaday/nmt-corpus/webnovel/train.raw.en").readlines()
zh_tmp = ""
en_tmp = ""
doc_idx = 0
for idx in range(len(zh)):
    o_len = len(zh_tmp) + len(en_tmp)
    n_len = len(zh[idx]) + len(en[idx])
    if o_len + n_len >= 4000:
        dataset.append({
            "input": zh_tmp.strip(),
            "output": en_tmp.strip()
        })
        zh_tmp = zh[idx]
        en_tmp = en[idx]
    else:
        zh_tmp += zh[idx]
        en_tmp += en[idx]
with open("translation_stack_2048.json", "w+", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)