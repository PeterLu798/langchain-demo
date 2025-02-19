from langchain.prompts import PromptTemplate

template = PromptTemplate.from_file("example_prompt_template.txt", encoding='utf-8')
print("===Template===")
print(template)
print("===Prompt===")
print(template.format(topic='黑色幽默'))