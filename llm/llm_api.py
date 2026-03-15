import asyncio

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv

load_dotenv(override=True)
def get_llm(model_name="qwen3.5:9b"):
    return ChatOllama(model=model_name,reasoning=False, base_url="http://localhost:11434", api_key="xx",verbose=True)

llm = get_llm()
query_llm_sys_prompt = ChatPromptTemplate.from_template(
        '你必须始终输出一个数字0或1。'
        "{question}"
    )
query_llm_chain = query_llm_sys_prompt | llm
async def query_llm(prompt, text):

    """
    prompt : list[str]
    text : input sentence

    return: predicted label
    """

    prompt_text = " ".join(prompt)

    full_prompt = prompt_text + "\n\n" + text

    response = await query_llm_chain.ainvoke({"question": full_prompt})
    # response = query_llm_chain.invoke({"question": full_prompt})

    return response.content


def improve_prompt(history,prompt):

    """
    prompt : list[str]

    return: improved prompt
    """

    sys_prompt = ChatPromptTemplate.from_template(
        "根据以往的优化记录优化以下提示，使大模型进行情感分析更加准确。\n"
        "请注意：\n"
        "1、仅返回优化后的prompt，不要输出其他额外内容\n"
        "2、仅在原有prompt上进行小幅度改动后续会进行多轮调整：\n"
        "优化记录：{history}\n"
        "prompt：{prompt}"
    )
    chain = sys_prompt | llm
        
    response = chain.invoke({"history": history,"prompt": prompt})
        
    return response.content


def agg_prompts(prompts):

    sys_prompt = ChatPromptTemplate.from_template(
        "请将以下提示优化为一个提示，使大模型进行情感分析更加准确。\n" 
        "请注意：\n"
        "1、仅返回优化后的prompt，不要输出其他额外内容\n"
        "2、仅在原有prompt上进行小幅度改动后续会进行多轮调整：\n"
        "prompts：{prompts}"
    )
    chain = sys_prompt | llm

    response = chain.invoke({"prompts": prompts})

    return response.content
        
# if __name__ == "__main__":
#     # prompt = ["判断用户输入的句子的情绪是否为正面，是则输出1，否则输出0"]
#     # text = "apparently reassembled from the cutting-room floor of any given daytime soap "
#     # print(query_llm(prompt, text))
#     print(improve_prompt([], "判断用户输入的句子的情绪是否为正面，是则输出1，否则输出0"))
   