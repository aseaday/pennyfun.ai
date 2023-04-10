import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
PROMPT_DICT = {
    "prompt_input": (
        "Translate the following chinese into English.\n"
        "### Input:\n{input}\n\n### Response:\n"
    ),
}
demo_input = """人潮涌动,各种叫卖声不绝,热闹的庙会之中,一个青衣少年顾盼生姿,充满好奇的四处张望!
突然他看到一个红衣少女,较好的容颜在雪白的狐狸围脖的映衬下,让少年看的如痴如醉,步步尾随被察觉后,那少女莞尔一笑故意扔下一个红色丝巾!
得到暗示后的青衣少年,很快就打听到红衣少女居然就住在隔壁村,二八年华还未曾婚配,欣喜若狂的少年忙托人求娶少女,偏偏女方家中索要巨额聘礼!
为了凑齐聘礼,男方家中卖掉了良田,卖掉了院落和家中的奴仆,三媒六聘热闹的把红衣少女娶进家门,一偿宿愿。"""
demo = PROMPT_DICT["prompt_input"].format(input=demo_input)
model = AutoModelForCausalLM.from_pretrained("./testx/checkpoint-1200", device_map={
    "": 0
}, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("./testx/checkpoint-1200")
model.eval()
inputs = tokenizer(demo, return_tensors="pt")
input_ids = inputs.input_ids.cuda()
generation_output = model.generate(
    input_ids=input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=1000
)
for s in generation_output.sequences:
    print(tokenizer.decode(s))