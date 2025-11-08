# Prompt-tuning experiment
from langchain.prompts import PromptTemplate
from langchain.llms import ChatOpenAI

prompt = PromptTemplate.from_template("Explain {concept} in simple terms.")
llm = ChatOpenAI(model="gpt-4o-mini")

for t in [0.2, 0.5, 0.8]:
    llm.temperature = t
    response = llm.invoke(prompt.format(concept="Agent Evaluation"))
    print(f"Temperature {t}: {response}")
