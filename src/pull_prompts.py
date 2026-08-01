"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# Prompt de baixa qualidade publicado pelo instrutor no Prompt Hub
SOURCE_PROMPT = "leonanluppi/bug_to_user_story_v1"

# Onde o prompt será salvo localmente
OUTPUT_PATH = "prompts/bug_to_user_story_v1.yml"

# Chave raiz usada dentro do YAML
PROMPT_KEY = "bug_to_user_story_v1"


def extract_template(message_template) -> str:
    """
    Extrai o texto bruto de um message template do LangChain.

    Cobre as variações de serialização (SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, mensagens já materializadas etc.).

    Args:
        message_template: Item de ChatPromptTemplate.messages

    Returns:
        Texto do template (string vazia se não for possível extrair)
    """
    inner = getattr(message_template, "prompt", None)

    if inner is not None:
        template = getattr(inner, "template", None)
        if isinstance(template, str):
            return template

    content = getattr(message_template, "content", None)
    if isinstance(content, str):
        return content

    return ""


def message_role(message_template) -> str:
    """
    Descobre o papel (system/human/...) de um message template.

    Args:
        message_template: Item de ChatPromptTemplate.messages

    Returns:
        Papel normalizado em minúsculas
    """
    role = getattr(message_template, "role", None)
    if isinstance(role, str) and role:
        return role.lower()

    class_name = type(message_template).__name__.lower()

    if "system" in class_name:
        return "system"
    if "human" in class_name or "user" in class_name:
        return "human"
    if "ai" in class_name or "assistant" in class_name:
        return "ai"

    return "human"


def prompt_to_dict(prompt) -> dict:
    """
    Converte um prompt do LangChain em um dicionário serializável em YAML.

    Args:
        prompt: ChatPromptTemplate (ou PromptTemplate) retornado por hub.pull

    Returns:
        Dicionário no mesmo formato usado pelos arquivos em prompts/
    """
    system_parts = []
    user_parts = []

    messages = getattr(prompt, "messages", None)

    if messages:
        for message in messages:
            text = extract_template(message)
            if not text:
                continue

            if message_role(message) == "system":
                system_parts.append(text)
            else:
                user_parts.append(text)
    else:
        # PromptTemplate simples (sem estrutura de chat)
        system_parts.append(getattr(prompt, "template", ""))

    return {
        "description": "Prompt de baixa qualidade puxado do LangSmith Prompt Hub",
        "system_prompt": "\n\n".join(system_parts).strip() + "\n",
        "user_prompt": "\n\n".join(user_parts).strip(),
        "version": "v1",
        "source": SOURCE_PROMPT,
        "input_variables": sorted(getattr(prompt, "input_variables", []) or []),
        "tags": ["bug-analysis", "user-story", "product-management"],
    }


def pull_prompts_from_langsmith() -> bool:
    """
    Faz pull do prompt v1 do LangSmith Hub e salva localmente em YAML.

    Returns:
        True se o prompt foi puxado e salvo com sucesso, False caso contrário
    """
    print(f"Puxando prompt do LangSmith Hub: {SOURCE_PROMPT}")

    try:
        prompt = hub.pull(SOURCE_PROMPT)
    except Exception as e:
        print(f"❌ Erro ao puxar '{SOURCE_PROMPT}': {e}\n")
        print("Verifique:")
        print("  - LANGSMITH_API_KEY está configurada corretamente no .env")
        print("  - O prompt existe e está público no LangSmith Hub")
        print("  - Sua conexão com a internet está funcionando")
        return False

    print("   ✓ Prompt carregado do Hub")

    prompt_data = prompt_to_dict(prompt)

    if not prompt_data["system_prompt"].strip():
        print("❌ O prompt puxado não contém conteúdo de system prompt.")
        return False

    variables = prompt_data["input_variables"]
    print(f"   ✓ Variáveis de entrada: {', '.join(variables) if variables else 'nenhuma'}")

    if not save_yaml({PROMPT_KEY: prompt_data}, OUTPUT_PATH):
        return False

    print(f"   ✓ Prompt salvo em: {OUTPUT_PATH}")
    return True


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    if not pull_prompts_from_langsmith():
        return 1

    print("\n✅ Pull concluído com sucesso!")
    print("\nPróximos passos:")
    print(f"1. Analise o prompt em {OUTPUT_PATH}")
    print("2. Crie sua versão otimizada em prompts/bug_to_user_story_v2.yml")
    print("3. Publique com: python src/push_prompts.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
