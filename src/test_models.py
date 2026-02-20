from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(path, prompt):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    print("🟢 Test Gemini")
    test_model("../models/gemini", "Hello LunarTech!")
    print("🟢 Test Kimi")
    test_model("../models/kimi", "Hello LunarTech!")
    print("🟢 Test GLM")
    test_model("../models/glm", "Hello LunarTech!")
