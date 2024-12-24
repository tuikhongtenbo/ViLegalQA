from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import regex
from typing import Tuple

def parse_answer(answer: str) -> Tuple[str, bool]:
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    json_content = pattern.findall(answer)[0]
    return json_content, True

def run_main(
    model_name: str = "vilm/vinallama-7b-chat",
    test_data_path: str = "/data/npl/MRC/ViLegalQA/Task1/test_data.json",
    output_path: str = "/data/npl/MRC/ViLegalQA/Task1/results_vinallama.json",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 256,
):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load test set
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Open output file and write results
    for idx, key in enumerate(tqdm(data, desc="Processing")):
        item = data[key]
        question = item.get("question", "")
        prompt = (
            f"{question} Câu trả lời cần ở dạng JSON, với khóa DOCUMENT chỉ loại văn bản pháp luật, "
            f"NUMBER chỉ số liệu của văn bản đó, CHAPTER chỉ chương, SECTION chỉ mục, SUBSECTION chỉ tiểu mục, "
            f"ARTICLE chỉ điều, CLAUSE chỉ khoản."
        )

        # Tokenize and generate output
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed_response, is_json = parse_answer(response)

        # Collect references
        reference = []
        contexts = item.get("contexts", {})
        for id, context in contexts.items():
            reference.append({
                "document": context.get("document"),
                "number": context.get("Số hiệu:"),
                "type": context.get("Loại văn bản:"),
                "article": context.get("Điều"),
                "clause": context.get("Khoản"),
            })

        # Prepare result
        result = {
            "question": question,
            "prompt": prompt,
            "is_response_in_json": is_json,
            "response": parsed_response,
            "reference": reference,
        }
        # Dumping results
        with open(output_path, 'a', encoding='utf-8') as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=4)
            output_file.write('\n')

if __name__ == "__main__":
    run_main()